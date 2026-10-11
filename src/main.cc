//命令行入口：
//  generate  加载 tiny 模型，按采样参数生成一段文字
//  run       终端多轮聊天（PromptBuilder + 流式输出）
//  inspect   只看 model.json 元数据
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/sampler.h"
#include "mini_llama/terminal.h"
#include "mini_llama/tokenizer.h"

using mini_llama::AsciiTokenizer;
using mini_llama::ChatMessage;
using mini_llama::ChatSession;
using mini_llama::ForwardBatch;
using mini_llama::InitTerminalUtf8;
using mini_llama::InspectModel;
using mini_llama::LoadModel;
using mini_llama::MiniBatch;
using mini_llama::MiniLlamaContext;
using mini_llama::MiniLlamaModel;
using mini_llama::MiniSampler;
using mini_llama::PromptBuilder;
using mini_llama::SamplingParams;
using mini_llama::Terminal;

//打印使用方法，包含 generate / run / inspect
static void PrintUsage(const char* argv0) {
  std::cout
      << "usage:\n"
      << "  " << argv0 << " generate [options]\n"
      << "  " << argv0 << " run [options]\n"
      << "  " << argv0 << " inspect [config]\n"
      << "\n"
      << "generate options:\n"
      << "  --config <path>      model.json 路径（默认 models/tiny/model.json）\n"
      << "  --model <path>       model.bin 路径（默认 models/tiny/model.bin）\n"
      << "  -p <text>            提示词（默认 hello）\n"
      << "  -n <N>               生成 token 数（默认 16）\n"
      << "  --temperature <T>    温度，0 表示 greedy\n"
      << "  --top-k <k>          只在分数最高的 k 个里抽，0 表示关闭\n"
      << "  --seed <S>           随机种子，0 表示不固定\n"
      << "\n"
      << "run options:\n"
      << "  --config <path>      model.json 路径（默认 models/tiny/model.json）\n"
      << "  --model <path>       model.bin 路径（默认 models/tiny/model.bin）\n"
      << "  -n <N>               每轮最多生成 token 数（默认 64）\n"
      << "  --temperature <T>    温度，0 表示 greedy\n"
      << "  --top-k <k>          top-k，0 表示关闭\n"
      << "  --seed <S>           随机种子，0 表示不固定\n"
      << "  --template <name>    聊天模板：plain（默认）或 qwen2\n"
      << "\n"
      << "inspect:\n"
      << "  只读 model.json，打印 config / tensor / tokenizer，不加载权重\n"
      << "  默认 config: models/tiny/model.json\n";
}

//只看清单，不读 bin
static int RunInspect(int argc, char** argv) {
  std::string config_path = "models/tiny/model.json";
  if (argc >= 3) {
    if (std::strcmp(argv[2], "-h") == 0 ||
        std::strcmp(argv[2], "--help") == 0) {
      PrintUsage(argv[0]);
      return 0;
    }
    config_path = argv[2];
  }
  return InspectModel(config_path) ? 0 : 1;
}

//解析浮点数参数
static bool ParseFloatArg(const char* text, float& out) {
  char* end = nullptr;
  const float value = std::strtof(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(value) || value < 0.0f) {
    return false;
  }
  out = value;
  return true;
}

//解析整数参数，10进制
static bool ParseIntArg(const char* text, int& out) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0 ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    return false;
  }
  out = static_cast<int>(value);
  return true;
}

//解析无符号整数参数
static bool ParseUintArg(const char* text, unsigned int& out) {
  char* end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0' ||
      value > static_cast<unsigned long>(
                   std::numeric_limits<unsigned int>::max())) {
    return false;
  }
  out = static_cast<unsigned int>(value);
  return true;
}

//两段token序列的公共前缀长度，用来决定 KV 能复用多少
static size_t CommonPrefixLength(const std::vector<int>& a,
                                 const std::vector<int>& b) {
  size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) {
    ++i;
  }
  return i;
}

//运行generate命令
static int RunGenerate(int argc, char** argv) {
  std::string config_path = "models/tiny/model.json";//默认模型配置路径
  std::string model_path = "models/tiny/model.bin";//默认模型权重路径
  std::string prompt = "hello";
  int n_predict = 16;//默认生成token数
  float temperature = 0.0f;
  int top_k = 0;//默认不启用top-k
  unsigned int seed = 0;//默认不固定随机种子

  //解析命令行参数
  for (int i = 2; i < argc; ++i) {
    //解析模型配置路径
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_path = argv[++i];
    } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
      prompt = argv[++i];
    } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_predict) || n_predict == 0) {
        std::cerr << "Invalid -n value. Expected a positive integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      if (!ParseFloatArg(argv[++i], temperature)) {
        std::cerr
            << "Invalid --temperature value. Expected a non-negative float.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], top_k)) {
        std::cerr
            << "Invalid --top-k value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      if (!ParseUintArg(argv[++i], seed)) {
        std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << argv[i] << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }

  //加载模型，tiny模型默认在models/tiny目录下
  MiniLlamaModel model = LoadModel(config_path, model_path);
  //判断模型是否加载成功
  if (!model.loaded) {
    std::cerr << "Failed to load model: " << model.load_error << "\n";
    return 1;
  }

  //创建ASCII编码器
  AsciiTokenizer tokenizer;
  //编码提示词
  std::vector<int> tokens = tokenizer.Encode(prompt);
  //判断提示词是否编码成功
  if (tokens.empty()) {
    std::cerr << "Prompt produced no tokens.\n";
    return 1;
  }

  SamplingParams sampling_params;
  sampling_params.temperature = temperature;
  sampling_params.top_k = top_k;
  sampling_params.seed = seed;
  MiniSampler sampler(sampling_params);

  //创建上下文
  MiniLlamaContext ctx(&model);
  //打印采样参数
  std::cout << "sampling: temperature=" << temperature << ", top_k=" << top_k
            << ", seed=" << seed << "\n";
  std::cout << "prompt: " << prompt << "\n";
  std::cout << tokenizer.Decode(tokens);
  //尝试生成文本
  try {
    //创建批量
    MiniBatch prefill = MiniBatch::FromTokens(tokens, 0);
    //前向传播
    auto logits = ForwardBatch(ctx, model, prefill);
    //采样下一个token
    int next = sampler.Sample(logits, sampling_params);
    //添加到tokens
    tokens.push_back(next);
    //打印token
    std::cout << tokenizer.DecodeToken(next) << std::flush;

    //生成剩余的token
    for (int i = 1; i < n_predict; ++i) {
      //判断是否到达EOS
      if (next == tokenizer.eos_id()) {
        break;
      }
      //创建批量
      MiniBatch decode = MiniBatch::Single(next, ctx.pos + 1);
      //前向传播
      logits = ForwardBatch(ctx, model, decode);
      next = sampler.Sample(logits, sampling_params);
      tokens.push_back(next);
      std::cout << tokenizer.DecodeToken(next) << std::flush;
    }
  } catch (const std::exception& e) {
    std::cerr << "\nError: " << e.what() << "\n";
    return 1;
  }

  std::cout << "\n";
  return 0;
}

//运行run命令，终端多轮聊天
static int RunChat(int argc, char** argv) {
  //如果需要打印帮助，打印使用方法
  if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                    std::strcmp(argv[2], "--help") == 0)) {
    PrintUsage(argv[0]);
    return 0;
  }

  std::string config_path = "models/tiny/model.json";
  std::string model_path = "models/tiny/model.bin";
  float temperature = 0.0f;
  int top_k = 0;
  unsigned int seed = 0;
  int max_response_tokens = 64;
  std::string chat_template;//空 = plain

  //解析命令行参数，从第3个参数开始
  for (int i = 2; i < argc; ++i) {
    //解析模型配置路径
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_path = argv[++i];
    } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      //解析模型权重路径
      model_path = argv[++i];
    } else if ((std::strcmp(argv[i], "-n") == 0 ||
                std::strcmp(argv[i], "--n-predict") == 0) &&
               i + 1 < argc) {
      //解析每轮最多生成token数
      if (!ParseIntArg(argv[++i], max_response_tokens) ||
          max_response_tokens == 0) {
        std::cerr
            << "Invalid --n-predict value. Expected a positive integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      //解析温度
      if (!ParseFloatArg(argv[++i], temperature)) {
        std::cerr
            << "Invalid --temperature value. Expected a non-negative float.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      //解析top-k
      if (!ParseIntArg(argv[++i], top_k)) {
        std::cerr
            << "Invalid --top-k value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      //解析随机种子
      if (!ParseUintArg(argv[++i], seed)) {
        std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--template") == 0 && i + 1 < argc) {
      //解析聊天模板，默认plain
      chat_template = argv[++i];
      //判断聊天模板是否有效，只能是plain或qwen2
      if (chat_template != "plain" && chat_template != "qwen2") {
        std::cerr << "Invalid --template. Expected plain or qwen2.\n";
        return 1;
      }
      //如果聊天模板是plain，清空
      if (chat_template == "plain") {
        chat_template.clear();
      }
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintUsage(argv[0]);
      return 0;
    } else {
      //如果参数无效，打印使用方法
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }

  //加载模型
  MiniLlamaModel model = LoadModel(config_path, model_path);
  //如果失败，打印错误信息
  if (!model.loaded) {
    std::cerr << "Failed to load model: " << model.load_error << "\n";
    return 1;
  }

  //创建ASCII编码器
  AsciiTokenizer tokenizer;
  //创建提示构建器
  PromptBuilder builder;
  //如果聊天模板有效，设置聊天模板
  if (!chat_template.empty()) {
    builder.SetChatTemplate(chat_template);
  }
  //创建终端
  Terminal term;
  //创建会话
  ChatSession session;
  //设置采样参数
  session.sampling_params.temperature = temperature;
  session.sampling_params.top_k = top_k;
  session.sampling_params.seed = seed;

  //plain模式在会话里放一句默认system
  if (chat_template.empty()) {
    session.AddMessage("system", "You are a helpful assistant.");
  }
  //打印聊天信息
  term.PrintMessage("mini-ai chat");
  term.PrintMessage("Type /help for commands, /exit to quit.\n");
  //测试模型，如果上下文窗口小于等于256，打印提示信息
  if (model.config.max_seq_len <= 256) {
    term.PrintMessage(
        "Tiny teaching model: random weights, small context window.");
    term.PrintMessage("This is a smoke-test chat loop, not a real LLM demo.\n");
  }

  MiniLlamaContext ctx(&model);

  while (true) {
    //打印用户提示符
    term.PrintUserPrompt();
    //读取输入
    std::string input = term.ReadLine();
    if (input.empty() && std::cin.eof()) {
      break;
    }

    if (input == "/help") {
      term.PrintHelp();
      continue;
    }
    if (input == "/exit") {
      break;
    }
    if (input == "/clear") {
      session.Clear();//清空会话
      ctx = MiniLlamaContext(&model);
      if (chat_template.empty()) {
        session.AddMessage("system", "You are a helpful assistant.");//如果聊天模板是plain，添加默认system
      }
      term.PrintMessage("Chat history cleared.\n");//打印清空信息
      continue;
    }
    if (input == "/stats") {
      term.PrintStats(session);
      continue;
    }
    if (input == "/params") {
      term.PrintParams(session.sampling_params);
      continue;
    }
    if (!input.empty() && input[0] == '/') {
      term.PrintMessage("Unknown command: " + input + "\n");
      continue;
    }
    if (input.empty()) {
      continue;
    }

    //先拼候选消息，算完再写回 session，避免超长失败时污染历史
    std::vector<ChatMessage> candidate_messages = session.messages;
    candidate_messages.push_back({"user", input});

    std::string prompt_text = builder.Build(candidate_messages);
    std::vector<int> tokens = tokenizer.Encode(prompt_text);

    if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
      term.PrintMessage(
          "Error: prompt uses " + std::to_string(tokens.size()) +
          " tokens, context window is " +
          std::to_string(model.config.max_seq_len) +
          ". Use /clear or a shorter prompt.\n");
      continue;
    }

    int max_response =
        model.config.max_seq_len - static_cast<int>(tokens.size());
    if (max_response > max_response_tokens) {
      max_response = max_response_tokens;
    }

    session.messages = candidate_messages;
    MiniSampler sampler(session.sampling_params);
    mini_llama::Tensor logits;

    auto start = std::chrono::steady_clock::now();

    //前缀复用，radix 命中和当前 KV 历史取较短的一段
    size_t cached_prefix_len = session.LongestCachedPrefix(tokens);
    size_t context_prefix_len = CommonPrefixLength(ctx.token_history, tokens);
    size_t prefix_len = std::min(cached_prefix_len, context_prefix_len);
    if (prefix_len >= tokens.size()) {
      ctx = MiniLlamaContext(&model);
      prefix_len = 0;
    } else if (prefix_len == 0) {
      ctx = MiniLlamaContext(&model);
    } else if (prefix_len < ctx.token_history.size()) {
      ctx.token_history.resize(prefix_len);
      ctx.pos = static_cast<int>(prefix_len - 1);
    }
    //截取新的提示token
    std::vector<int> new_prompt_tokens(
        tokens.begin() + static_cast<std::ptrdiff_t>(prefix_len), tokens.end());
    session.SetTokenHistory(tokens);

    //只prefill还没进KV的那一段，只做一次前向传播
    {
      MiniBatch prefill = MiniBatch::FromTokens(new_prompt_tokens,
                                                static_cast<int>(prefix_len));
      logits = ForwardBatch(ctx, model, prefill);
    }

    //逐个token采样，边解码边打到终端
    std::vector<int> generated_ids;
    std::string streamed_reply;
    int generated_count = 0;

    term.PrintAssistantPrefix();
    try {
      for (int i = 0; i < max_response; ++i) {
        int next_token = sampler.Sample(logits, session.sampling_params);
        tokens.push_back(next_token);
        session.AppendToken(next_token);
        ++generated_count;

        if (next_token != tokenizer.eos_id()) {
          generated_ids.push_back(next_token);
          std::string current_reply = tokenizer.Decode(generated_ids);
          if (current_reply.size() > streamed_reply.size()) {
            term.PrintTokenText(current_reply.substr(streamed_reply.size()));
            term.Flush();
            streamed_reply = current_reply;
          }
        }

        //创建解码批量
        MiniBatch decode_batch =
            MiniBatch::Single(next_token, static_cast<int>(tokens.size() - 1));
        logits = ForwardBatch(ctx, model, decode_batch);
        if (next_token == tokenizer.eos_id()) {
          break;
        }
      }
    } catch (const std::exception& e) {
      term.NewLine();
      term.PrintMessage(std::string("Inference error: ") + e.what() + "\n");
      continue;
    }

    //解码生成回复
    std::string assistant_reply = tokenizer.Decode(generated_ids);
    //如果回复长度大于流式回复，打印新的部分
    if (assistant_reply.size() > streamed_reply.size()) {
      term.PrintTokenText(assistant_reply.substr(streamed_reply.size()));
    }
    term.NewLine();
    term.NewLine();

    auto end = std::chrono::steady_clock::now();//记录结束时间
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();//计算耗时

    session.AddMessage("assistant", assistant_reply);//添加回复消息
    session.RecordTurn(static_cast<int>(new_prompt_tokens.size()),
                       generated_count, elapsed_ms);//记录回合
    session.RecordPrefix(session.token_history);//记录前缀
  }

  term.PrintMessage("Goodbye.\n");
  return 0;
}

int main(int argc, char** argv) {
  InitTerminalUtf8();

  if (argc < 2 || std::strcmp(argv[1], "help") == 0 ||
      std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
    PrintUsage(argv[0]);
    return 0;
  }

  if (std::strcmp(argv[1], "generate") == 0) {//运行generate命令
    return RunGenerate(argc, argv);
  }
  if (std::strcmp(argv[1], "run") == 0) {
    return RunChat(argc, argv);
  }
  if (std::strcmp(argv[1], "inspect") == 0) {
    return RunInspect(argc, argv);
  }

  std::cerr << "Unknown command: " << argv[1]
            << "\nExpected: generate, run, or inspect\n";
  PrintUsage(argv[0]);
  return 1;
}
