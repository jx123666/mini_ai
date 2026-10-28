//把多轮消息拼成prompt
//空模板走 System/User/Assistant；"qwen2"走内置格式
//以{%或{{开头的字符串按GGUF里的Jinja聊天模板执行
#include "mini_llama/prompt_builder.h"

#include <cctype>
#include <stack>
#include <string>
#include <vector>
#include <stdexcept>

#include "mini_llama/gguf.h"

namespace mini_llama {
namespace {

struct EvalContext {
  const std::vector<ChatMessage>* messages = nullptr;
  const ChatMessage* current_message = nullptr;
  size_t message_index = 0;
  bool add_generation_prompt = false;
};

std::string Trim(const std::string& s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) {
    ++a;
  }
  size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) {
    --b;
  }
  return s.substr(a, b - a);
}

// 去掉标签内部的空白控制标记，标签外的空白在编译时处理。
std::string StripJinjaTrimMarkers(const std::string& raw) {
  std::string value = Trim(raw);
  if (!value.empty() && value.front() == '-') {
    value = Trim(value.substr(1));
  }
  if (!value.empty() && value.back() == '-') {
    value.pop_back();
    value = Trim(value);
  }
  return value;
}

//算一个表达式：字符串、变量、message['role']、用+拼起来
std::string EvalExpr(const std::string& expr_raw, const EvalContext& ctx) {
  std::string expr = Trim(expr_raw);
  std::vector<std::string> parts;
  size_t pos = 0;
  while (pos < expr.size()) {
    while (pos < expr.size() &&
           std::isspace(static_cast<unsigned char>(expr[pos]))) {
      ++pos;
    }
    if (pos >= expr.size()) {
      break;
    }
    if (expr[pos] == '\'' || expr[pos] == '"') {
      const char quote = expr[pos++];
      std::string lit;
      while (pos < expr.size() && expr[pos] != quote) {
        char ch = expr[pos++];
        if (ch == '\\') {
          if (pos == expr.size()) throw std::runtime_error("unfinished template escape");
          ch = expr[pos++];
          if (ch == 'n') ch = '\n';
          else if (ch == 'r') ch = '\r';
          else if (ch == 't') ch = '\t';
          else if (ch != '\\' && ch != '\'' && ch != '"')
            throw std::runtime_error("unsupported template escape");
        }
        lit += ch;
      }
      if (pos == expr.size()) throw std::runtime_error("unterminated template string");
      ++pos;
      parts.push_back(lit);
    } else {
      size_t start = pos;
      while (pos < expr.size() && expr[pos] != '+' && expr[pos] != '\'') {
        if (expr[pos] == '[') {
          ++pos;
          while (pos < expr.size() && expr[pos] != ']') {
            ++pos;
          }
          if (pos < expr.size()) {
            ++pos;
          }
        } else if (expr[pos] == '.') {
          ++pos;
          while (pos < expr.size() &&
                 !std::isspace(static_cast<unsigned char>(expr[pos])) &&
                 expr[pos] != '+' && expr[pos] != '\'' && expr[pos] != '[' &&
                 expr[pos] != '.') {
            ++pos;
          }
        } else {
          ++pos;
        }
      }
      std::string token = Trim(expr.substr(start, pos - start));
      if (token == "true" || token == "True" || token == "false" || token == "False") {
        parts.push_back(token);
      } else if (token == "add_generation_prompt") {
        parts.push_back(ctx.add_generation_prompt ? "true" : "false");
      } else if (token == "message['role']" || token == "message[\"role\"]") {
        parts.push_back(ctx.current_message ? ctx.current_message->role : "");
      } else if (token == "message['content']" ||
                 token == "message[\"content\"]") {
        parts.push_back(ctx.current_message ? ctx.current_message->content
                                            : "");
      } else if (token == "loop.first") {
        parts.push_back(ctx.message_index == 0 ? "true" : "false");
      } else if (token.rfind("messages[", 0) == 0) {
        size_t close = token.find(']');
        std::string key;
        if (close != std::string::npos && close + 1 < token.size() &&
            token[close + 1] == '[') {
          size_t key_start = close + 2;
          size_t key_end = token.find(']', key_start);
          if (key_end != std::string::npos) {
            key = token.substr(key_start, key_end - key_start);
            if (key.size() >= 2 &&
                ((key.front() == '\'' && key.back() == '\'') ||
                 (key.front() == '"' && key.back() == '"'))) {
              key = key.substr(1, key.size() - 2);
            }
          }
          try {
            int idx = std::stoi(Trim(token.substr(9, close - 9)));
            if (ctx.messages && idx >= 0 &&
                idx < static_cast<int>(ctx.messages->size())) {
              const ChatMessage& msg = (*ctx.messages)[static_cast<size_t>(idx)];
              if (key == "role") {
                parts.push_back(msg.role);
              } else if (key == "content") {
                parts.push_back(msg.content);
              }
            }
          } catch (...) {
          }
        }
      } else throw std::runtime_error("unsupported template expression: " + token);
    }
    while (pos < expr.size() &&
           std::isspace(static_cast<unsigned char>(expr[pos]))) {
      ++pos;
    }
    if (pos < expr.size() && expr[pos] == '+') {
      ++pos;
    }
  }
  std::string result;
  for (const auto& part : parts) {
    result += part;
  }
  return result;
}

bool EvalCondition(const std::string& cond_raw, const EvalContext& ctx) {
  std::string cond = Trim(cond_raw);
  if (cond.empty()) {
    return false;
  }
  size_t or_pos = cond.find(" or ");
  if (or_pos != std::string::npos) {
    return EvalCondition(cond.substr(0, or_pos), ctx) ||
           EvalCondition(cond.substr(or_pos + 4), ctx);
  }
  size_t and_pos = cond.find(" and ");
  if (and_pos != std::string::npos) {
    return EvalCondition(cond.substr(0, and_pos), ctx) &&
           EvalCondition(cond.substr(and_pos + 5), ctx);
  }
  if (cond.rfind("not ", 0) == 0) return !EvalCondition(cond.substr(4), ctx);
  size_t neq_pos = cond.find(" != ");
  if (neq_pos != std::string::npos) {
    return EvalExpr(cond.substr(0, neq_pos), ctx) !=
           EvalExpr(cond.substr(neq_pos + 4), ctx);
  }
  size_t eq_pos = cond.find(" == ");
  if (eq_pos != std::string::npos) {
    return EvalExpr(cond.substr(0, eq_pos), ctx) ==
           EvalExpr(cond.substr(eq_pos + 4), ctx);
  }
  std::string val = EvalExpr(cond, ctx);
  if (val == "true" || val == "True") {
    return true;
  }
  if (val == "false" || val == "False" || val.empty()) {
    return false;
  }
  return true;
}

//模板编译成几条指令，if/for用jump跳过或回到循环头
enum class InstrType { kText, kOutput, kIf, kEndIf, kFor, kEndFor };

struct Instr {
  InstrType type = InstrType::kText;
  std::string text;
  size_t jump = 0;
};

struct ForFrame {
  size_t pc = 0;
  size_t index = 0;
};

std::vector<Instr> CompileTemplate(const std::string& tmpl) {
  std::vector<Instr> instrs;
  std::stack<size_t> if_stack;
  std::stack<size_t> for_stack;
  std::vector<char> blocks;
  size_t pos = 0;
  while (pos < tmpl.size()) {
    const bool trim_left = pos + 2 < tmpl.size() && tmpl[pos] == '{' &&
        (tmpl[pos + 1] == '%' || tmpl[pos + 1] == '{') && tmpl[pos + 2] == '-';
    if (trim_left && !instrs.empty() && instrs.back().type == InstrType::kText) {
      auto& text = instrs.back().text;
      while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.pop_back();
    }
    if (pos + 1 < tmpl.size() && tmpl[pos] == '{' && tmpl[pos + 1] == '%') {
      pos += 2;
      size_t end = tmpl.find("%}", pos);
      if (end == std::string::npos) {
        throw std::runtime_error("unterminated template statement");
      }
      std::string stmt = StripJinjaTrimMarkers(tmpl.substr(pos, end - pos));
      pos = end + 2;
      if (end > 0 && tmpl[end - 1] == '-')
        while (pos < tmpl.size() && std::isspace(static_cast<unsigned char>(tmpl[pos]))) ++pos;
      if (stmt.rfind("for ", 0) == 0) {
        if (stmt != "for message in messages" || !for_stack.empty())
          throw std::runtime_error("unsupported template loop: " + stmt);
        instrs.push_back({InstrType::kFor, stmt, 0});
        for_stack.push(instrs.size() - 1);
        blocks.push_back('f');
      } else if (stmt == "endfor") {
        if (for_stack.empty() || blocks.empty() || blocks.back() != 'f') {
          throw std::runtime_error("unexpected endfor");
        }
        size_t for_pc = for_stack.top();
        for_stack.pop();
        blocks.pop_back();
        instrs.push_back({InstrType::kEndFor, "", for_pc});
        instrs[for_pc].jump = instrs.size();
      } else if (stmt.rfind("if ", 0) == 0) {
        instrs.push_back({InstrType::kIf, stmt.substr(3), 0});
        if_stack.push(instrs.size() - 1);
        blocks.push_back('i');
      } else if (stmt == "endif") {
        if (if_stack.empty() || blocks.empty() || blocks.back() != 'i') {
          throw std::runtime_error("unexpected endif");
        }
        size_t if_pc = if_stack.top();
        if_stack.pop();
        blocks.pop_back();
        instrs.push_back({InstrType::kEndIf, "", 0});
        instrs[if_pc].jump = instrs.size();
      } else throw std::runtime_error("unsupported template statement: " + stmt);
    } else if (pos + 1 < tmpl.size() && tmpl[pos] == '{' &&
               tmpl[pos + 1] == '{') {
      pos += 2;
      size_t end = tmpl.find("}}", pos);
      if (end == std::string::npos) {
        throw std::runtime_error("unterminated template expression");
      }
      std::string expr = StripJinjaTrimMarkers(tmpl.substr(pos, end - pos));
      pos = end + 2;
      if (end > 0 && tmpl[end - 1] == '-')
        while (pos < tmpl.size() && std::isspace(static_cast<unsigned char>(tmpl[pos]))) ++pos;
      instrs.push_back({InstrType::kOutput, expr, 0});
    } else {
      size_t end = pos;
      while (end < tmpl.size() &&
             !(tmpl[end] == '{' && end + 1 < tmpl.size() &&
               (tmpl[end + 1] == '%' || tmpl[end + 1] == '{'))) {
        ++end;
      }
      std::string text = tmpl.substr(pos, end - pos);
      pos = end;
      if (!text.empty()) {
        instrs.push_back({InstrType::kText, text, 0});
      }
    }
  }
  if (!if_stack.empty() || !for_stack.empty()) throw std::runtime_error("unclosed template block");
  return instrs;
}

std::string ExecuteTemplate(const std::vector<Instr>& instrs,
                            const std::vector<ChatMessage>& messages,
                            bool add_gen) {
  std::string output;
  EvalContext ctx;
  ctx.messages = &messages;
  ctx.add_generation_prompt = add_gen;
  std::stack<ForFrame> for_stack;
  size_t pc = 0;
  while (pc < instrs.size()) {
    const Instr& instr = instrs[pc];
    switch (instr.type) {
      case InstrType::kText:
        output += instr.text;
        ++pc;
        break;
      case InstrType::kOutput:
        output += EvalExpr(instr.text, ctx);
        ++pc;
        break;
      case InstrType::kIf:
        pc = EvalCondition(instr.text, ctx) ? pc + 1 : instr.jump;
        break;
      case InstrType::kEndIf:
        ++pc;
        break;
      case InstrType::kFor: {
        if (for_stack.empty() || for_stack.top().pc != pc) {
          if (messages.empty()) {
            pc = instr.jump;
          } else {
            ForFrame frame;
            frame.pc = pc;
            frame.index = 0;
            for_stack.push(frame);
            ctx.current_message = &messages[0];
            ctx.message_index = 0;
            ++pc;
          }
        } else {
          ForFrame& frame = for_stack.top();
          ++frame.index;
          if (frame.index >= messages.size()) {
            for_stack.pop();
            pc = instr.jump;
          } else {
            ctx.current_message = &messages[frame.index];
            ctx.message_index = frame.index;
            ++pc;
          }
        }
        break;
      }
      case InstrType::kEndFor:
        pc = instr.jump;
        break;
    }
  }
  return output;
}

}

void PromptBuilder::SetChatTemplate(const std::string& template_str) {
  chat_template_ = template_str;
}

std::string PromptBuilder::Build(
    const std::vector<ChatMessage>& messages) const {
  if (chat_template_.empty()) {
    return BuildPlain(messages);
  }
  //GGUF里存的是Jinja原文，不是"qwen2"这个名字
  if (chat_template_.find("{%") != std::string::npos ||
      chat_template_.find("{{") != std::string::npos) {
    return ExecuteTemplate(CompileTemplate(chat_template_), messages, true);
  }
  if (chat_template_ == "qwen2") {
    return BuildQwen2(messages);
  }
  return BuildPlain(messages);
}

//纯文本格式：System:/User:/Assistant，最后补上 Assistant，让模型续写
std::string PromptBuilder::BuildPlain(
    const std::vector<ChatMessage>& messages) const {
  std::string prompt;
  for (const auto& msg : messages) {
    if (msg.role == "system") {
      prompt += "System: " + msg.content + "\n";
    } else if (msg.role == "user") {
      prompt += "User: " + msg.content + "\n";
    } else if (msg.role == "assistant") {
      prompt += "Assistant: " + msg.content + "\n";
    }
    //不认识的role直接跳过
  }
  prompt += "Assistant:";
  return prompt;
}

//Qwen2常见模板，每条消息包在 <|im_start|>...<|im_end|>里
//没有system就补一句默认提示词
std::string PromptBuilder::BuildQwen2(
    const std::vector<ChatMessage>& messages) const {
  std::string prompt;
  bool has_system = false;
  for (const auto& msg : messages) {
    if (msg.role == "system") {
      has_system = true;
      break;
    }
  }
  if (!has_system) {
    prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
  }

  for (const auto& msg : messages) {
    if (msg.role != "system" && msg.role != "user" && msg.role != "assistant") {
      continue;
    }
    prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
  }
  prompt += "<|im_start|>assistant\n";
  return prompt;
}

//读tokenizer.chat_template；没有时qwen2架构退回内置名，别的架构返回空
std::string LoadChatTemplateFromGguf(const std::string& gguf_path) {
  std::string tmpl;
  if (GgufGetMetadataString(gguf_path, "tokenizer.chat_template", tmpl)) {
    return tmpl;
  }
  std::string architecture;
  if (GgufGetMetadataString(gguf_path, "general.architecture", architecture) &&
      architecture == "qwen2") {
    return "qwen2";
  }
  return "";
}

}
