#include "mini_llama/prompt_builder.h"

namespace mini_llama {

void PromptBuilder::SetChatTemplate(const std::string& template_str) {
  chat_template_ = template_str;
}

std::string PromptBuilder::Build(
    const std::vector<ChatMessage>& messages) const {
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
  //遍历消息，判断是否存在system角色
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
  //补上assistant角色，让模型续写
  prompt += "<|im_start|>assistant\n";
  return prompt;
}

}
