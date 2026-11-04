//命令行入口：
//  generate  加载模型并生成文本
//  bench     统计 prefill 和 decode 吞吐量
//  run       终端多轮聊天（PromptBuilder + 流式输出）
//  inspect       只看 model.json 元数据
//  inspect-gguf  只看 GGUF 的 metadata / tensor 名字
//  tokenize-gguf 用GGUF里的BPE词表编码再解码
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/runtime.h"
#include "mini_llama/backend.h"
#include "mini_llama/debug.h"
#include "mini_llama/request_context.h"
#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/loader.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/sampler.h"
#include "mini_llama/terminal.h"
#include "mini_llama/tokenizer.h"

using mini_llama::AsciiTokenizer;
using mini_llama::ChatMessage;
using mini_llama::ChatSession;
using mini_llama::CreateBpeTokenizer;
using mini_llama::CreateGgufTokenizer;
using mini_llama::ForwardBatch;
using mini_llama::GgufReader;
using mini_llama::ITokenizer;
using mini_llama::InitTerminalUtf8;
using mini_llama::InspectGguf;
using mini_llama::InspectModel;
using mini_llama::LoadChatTemplateFromGguf;
using mini_llama::LoadGgufModel;
using mini_llama::LoadModel;
using mini_llama::MiniBatch;
using mini_llama::MiniLlamaContext;
using mini_llama::MiniLlamaModel;
using mini_llama::MiniSampler;
using mini_llama::PromptBuilder;
using mini_llama::SamplingParams;
using mini_llama::Terminal;

//打印使用方法，包含 generate / run / inspect / inspect-gguf
static void PrintUsage(const char* argv0) {
  std::cout
      << "usage:\n"
      << "  " << argv0 << " generate [options]\n"
      << "  " << argv0 << " bench [model_dir_or_gguf] [options]\n"
      << "  " << argv0 << " run [model_dir_or_gguf] [options]\n"
      << "  " << argv0 << " inspect [config]\n"
      << "  " << argv0 << " inspect-gguf <path>\n"
      << "  " << argv0 << " tokenize-gguf <path> <text>\n"
      << "\n"
      << "generate / bench options:\n"
      << "  --config <path>      model.json 路径（默认 models/tiny/model.json）\n"
      << "  --model <path>       model.bin 路径（默认 models/tiny/model.bin）\n"
      << "  -p <text>            提示词（默认 hello）\n"
      << "  -n <N>               生成 token 数（默认 16）\n"
      << "  --temperature <T>    温度，0 表示 greedy\n"
      << "  --top-k <k>          只在分数最高的 k 个里抽，0 表示关闭\n"
      << "  --seed <S>           随机种子，0 表示不固定\n"
      << "  --threads <N>        CPU 线程数，0 自动选择\n"
      << "  --matmul <mode>      naive/threaded/simd/threaded-simd\n"
      << "  --quant <type>       q8_0/q4_0/q4_1\n"
      << "  --dump-logits <dir>  保存各步 logits 和 token 序列\n"
      << "  --tokenizer <path>   显式指定 JSON 词表\n"
      << "  --verbose           输出位置和 logits top-5\n"
      << "  --backend <name>     cpu/cuda\n"
      << "  --device <N>         CUDA 设备编号\n"
      << "\n"
      << "run:\n"
      << "  位置参数可以是目录或 .gguf。目录里有 .gguf 且没有 model.json 时走真实模型。\n"
      << "  例: " << argv0 << " run models/chat -n 32 --temperature 0.7\n"
      << "run options:\n"
      << "  --config <path>      model.json 路径（默认 models/tiny/model.json）\n"
      << "  --model <path>       model.bin 路径（默认 models/tiny/model.bin）\n"
      << "  -n <N>               每轮最多生成 token 数（默认 64）\n"
      << "  --temperature <T>    温度，0 表示 greedy\n"
      << "  --top-k <k>          top-k，0 表示关闭\n"
      << "  --seed <S>           随机种子，0 表示不固定\n"
      << "  --template <name>    plain、qwen2，或一份带聊天模板的 .gguf\n"
      << "\n"
      << "inspect:\n"
      << "  只读 model.json，打印 config / tensor / tokenizer，不加载权重\n"
      << "  默认 config: models/tiny/model.json\n"
      << "\n"
      << "inspect-gguf:\n"
      << "  读 GGUF，打印 metadata 和 tensor 名字（不映射成模型）\n"
      << "\n"
      << "tokenize-gguf:\n"
      << "  用 GGUF 里的 BPE 词表编码文本，并解码回来核对\n";
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

//只解析 GGUF 结构并打印
static int RunInspectGguf(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " inspect-gguf <path>\n";
    return 1;
  }
  if (std::strcmp(argv[2], "-h") == 0 || std::strcmp(argv[2], "--help") == 0) {
    PrintUsage(argv[0]);
    return 0;
  }
  GgufReader reader;
  if (!reader.Load(argv[2])) {
    std::cerr << "Failed to load GGUF: " << reader.load_error << "\n";
    return 1;
  }
  InspectGguf(reader);
  return 0;
}

//从GGUF读词表，打印id和roundtrip
static int RunTokenizeGguf(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " tokenize-gguf <path> <text>\n";
    return 1;
  }
  auto tokenizer = CreateGgufTokenizer(argv[2]);
  if (!tokenizer) {
    std::cerr << "Failed to load GGUF tokenizer\n";
    return 1;
  }
  const std::string text = argv[3];
  const std::vector<int> ids = tokenizer->Encode(text);
  std::cout << "bos=" << tokenizer->bos_id() << " eos=" << tokenizer->eos_id()
            << " unk=" << tokenizer->unk_id() << "\n";
  std::cout << "ids:";
  for (int id : ids) {
    std::cout << " " << id;
  }
  std::cout << "\n";
  std::cout << "decode: " << tokenizer->Decode(ids) << "\n";
  return 0;
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
  errno = 0;
  const long value = std::strtol(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' || value < 0 ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    return false;
  }
  out = static_cast<int>(value);
  return true;
}

//解析无符号整数参数
static bool ParseUintArg(const char* text, unsigned int& out) {
  char* end = nullptr;
  if (text[0] == '-') return false;
  errno = 0;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' ||
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

// 生成和基准测试共用同一条推理路径，计时不包含终端输出。
static int RunGenerate(int argc, char** argv, bool bench = false) {
  std::string config_path = "models/tiny/model.json";
  std::string model_path = "models/tiny/model.bin";
  std::string prompt = "hello", dump_dir, tokenizer_path, quant, mode = "threaded-simd";
  int n_predict = bench ? 64 : 16, threads = 0, device = 0;
  std::string backend = "cpu";
  bool verbose = false;
  SamplingParams params;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> const char* {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--config") config_path = value();
    else if (arg == "--model") model_path = value();
    else if (arg == "-p" || arg == "--prompt") prompt = value();
    else if (arg == "-n" || arg == "--n-predict") {
      if (!ParseIntArg(value(), n_predict) || n_predict == 0) throw std::runtime_error("invalid token count");
    } else if (arg == "--temperature") {
      if (!ParseFloatArg(value(), params.temperature)) throw std::runtime_error("invalid temperature");
    } else if (arg == "--top-k") {
      if (!ParseIntArg(value(), params.top_k)) throw std::runtime_error("invalid top-k");
    } else if (arg == "--seed") {
      if (!ParseUintArg(value(), params.seed)) throw std::runtime_error("invalid seed");
    } else if (arg == "--threads") {
      if (!ParseIntArg(value(), threads) || threads > 256) throw std::runtime_error("invalid thread count");
    } else if (arg == "--device") {
      if (!ParseIntArg(value(), device)) throw std::runtime_error("invalid device id");
    } else if (arg == "--backend") backend = value();
    else if (arg == "--matmul") mode = value();
    else if (arg == "--quant") quant = value();
    else if (arg == "--dump-logits") dump_dir = value();
    else if (arg == "--tokenizer") tokenizer_path = value();
    else if (arg == "--verbose") verbose = true;
    else if (arg == "-h" || arg == "--help") { PrintUsage(argv[0]); return 0; }
    else if (bench && arg.front() != '-') model_path = arg;
    else throw std::runtime_error("unknown option: " + arg);
  }
  mini_llama::SetMatmulThreads(static_cast<unsigned>(threads));
  using mini_llama::MatmulMode;
  if (mode == "naive") mini_llama::SetDefaultMatmulMode(MatmulMode::kNaive);
  else if (mode == "threaded") mini_llama::SetDefaultMatmulMode(MatmulMode::kThreaded);
  else if (mode == "simd") mini_llama::SetDefaultMatmulMode(MatmulMode::kSimd);
  else if (mode == "threaded-simd") mini_llama::SetDefaultMatmulMode(MatmulMode::kThreadedSimd);
  else throw std::runtime_error("unknown matmul mode: " + mode);

  auto request = mini_llama::StartRequest(bench ? "bench" : "generate", backend, model_path);
  auto start = mini_llama::RequestClock::now();
  auto loaded = mini_llama::LoadInferenceModel(model_path, config_path, tokenizer_path);
  auto& model = loaded.model;
  auto& tokenizer = *loaded.tokenizer;
  if (!quant.empty()) {
    if (quant == "q8_0") mini_llama::QuantizeModel(model, mini_llama::QuantType::kQ80);
    else if (quant == "q4_0") mini_llama::QuantizeModel(model, mini_llama::QuantType::kQ40);
    else if (quant == "q4_1") mini_llama::QuantizeModel(model, mini_llama::QuantType::kQ41);
    else throw std::runtime_error("unsupported quantization: " + quant);
  }
  if (backend == "cuda") mini_llama::EnableCuda(model, device);
  else if (backend != "cpu") throw std::runtime_error("unknown backend: " + backend);
  request.model_load_ms = mini_llama::ElapsedMs(start);
  start = mini_llama::RequestClock::now();
  const auto tokens = tokenizer.Encode(prompt);
  request.tokenize_ms = mini_llama::ElapsedMs(start);
  if (tokens.empty()) throw std::runtime_error("prompt produced no tokens");
  if (tokens.size() > static_cast<size_t>(model.config.max_seq_len) ||
      static_cast<size_t>(n_predict - 1) > static_cast<size_t>(model.config.max_seq_len) - tokens.size())
    throw std::runtime_error("prompt and generation exceed context window");
  request.prompt_tokens = request.prefill_tokens = static_cast<int>(tokens.size());
  MiniLlamaContext ctx(&model);
  MiniSampler sampler(params);
  start = mini_llama::RequestClock::now();
  auto logits = ForwardBatch(ctx, model, MiniBatch::FromTokens(tokens, 0));
  request.prefill_ms = mini_llama::ElapsedMs(start);
  std::vector<int> generated;
  std::string streamed;
  for (int step = 0; step < n_predict; ++step) {
    mini_llama::DumpLogits(dump_dir, step, logits);
    if (verbose) std::cout << "step=" << step << " pos=" << ctx.pos
                           << " logits=" << mini_llama::FormatTopLogits(logits) << '\n';
    start = mini_llama::RequestClock::now();
    const int next = sampler.Sample(logits, params);
    request.sample_ms += mini_llama::ElapsedMs(start);
    generated.push_back(next);
    if (!bench && next != tokenizer.eos_id()) {
      // 按完整 token 序列解码，避免把跨 token 的 UTF-8 字节拆坏。
      const std::string decoded = tokenizer.Decode(generated);
      if (decoded.size() > streamed.size()) std::cout << decoded.substr(streamed.size()) << std::flush;
      streamed = decoded;
    }
    if (next == tokenizer.eos_id() || step + 1 == n_predict) break;
    start = mini_llama::RequestClock::now();
    logits = ForwardBatch(ctx, model, MiniBatch::Single(next, ctx.pos + 1));
    request.decode_ms += mini_llama::ElapsedMs(start);
    ++request.decode_tokens;
  }
  request.generated_tokens = static_cast<int>(generated.size());
  request.Finish();
  if (!dump_dir.empty()) {
    std::ofstream metadata(std::filesystem::path(dump_dir) / "tokens.json");
    metadata << "{\"prompt\":[";
    for (size_t i = 0; i < tokens.size(); ++i) metadata << (i ? "," : "") << tokens[i];
    metadata << "],\"generated\":[";
    for (size_t i = 0; i < generated.size(); ++i) metadata << (i ? "," : "") << generated[i];
    metadata << "]}\n";
    if (!metadata) throw std::runtime_error("failed to write token metadata");
  }
  std::cout << '\n' << mini_llama::FormatRequestTraceSummary(request) << '\n';
  if (model.cuda) std::cout << mini_llama::CudaStats(model, ctx) << '\n';
  if (bench) std::cout << "prefill_tokens_per_second="
      << mini_llama::TokensPerSecond(request.prefill_tokens, request.prefill_ms)
      << " decode_tokens_per_second="
      << mini_llama::TokensPerSecond(request.decode_tokens, request.decode_ms)
      << " threads=" << mini_llama::MatmulThreads() << " matmul=" << mode << '\n';
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
  std::string gguf_path;
  std::string model_source;
  float temperature = 0.0f;
  int top_k = 0;
  unsigned int seed = 0;
  int max_response_tokens = 64;
  std::string chat_template;//空 = plain
  bool template_set = false;
  bool model_flag = false;
  int threads = 0;
  std::string quant, tokenizer_path, backend = "cpu";
  int device = 0;

  //run models/chat：第一个不是选项的参数是模型目录或 gguf 文件
  int arg_begin = 2;
  if (argc >= 3 && argv[2][0] != '-') {
    model_source = argv[2];
    arg_begin = 3;
  }

  //解析命令行参数
  for (int i = arg_begin; i < argc; ++i) {
    //解析模型配置路径
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_path = argv[++i];
    } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      //解析模型权重路径；显式指定后不再走目录里的 gguf
      model_path = argv[++i];
      model_flag = true;
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
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], threads) || threads > 256) throw std::runtime_error("invalid thread count");
    } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      backend = argv[++i];
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], device)) throw std::runtime_error("invalid device id");
    } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
      quant = argv[++i];
    } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (std::strcmp(argv[i], "--template") == 0 && i + 1 < argc) {
      //解析聊天模板，默认plain
      chat_template = argv[++i];
      template_set = true;
      //plain清空；.gguf则读出里面的tokenizer.chat_template
      if (chat_template == "plain") {
        chat_template.clear();
      } else if (chat_template.size() >= 5 &&
                 chat_template.substr(chat_template.size() - 5) == ".gguf") {
        chat_template = LoadChatTemplateFromGguf(chat_template);
      } else if (chat_template != "qwen2") {
        std::cerr << "Invalid --template. Expected plain, qwen2, or a .gguf path.\n";
        return 1;
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

  const std::string source = !model_flag && !model_source.empty() ? model_source : model_path;
  mini_llama::SetMatmulThreads(static_cast<unsigned>(threads));
  auto loaded = mini_llama::LoadInferenceModel(source, config_path, tokenizer_path);
  auto& model = loaded.model;
  ITokenizer& tokenizer = *loaded.tokenizer;
  if (!template_set) chat_template = loaded.chat_template;
  if (std::filesystem::path(loaded.source).extension() == ".gguf") gguf_path = loaded.source;
  if (!quant.empty()) {
    if (quant == "q8_0") mini_llama::QuantizeModel(model, mini_llama::QuantType::kQ80);
    else if (quant == "q4_0") mini_llama::QuantizeModel(model, mini_llama::QuantType::kQ40);
    else if (quant == "q4_1") mini_llama::QuantizeModel(model, mini_llama::QuantType::kQ41);
    else throw std::runtime_error("unsupported quantization: " + quant);
  }

  if (backend == "cuda") mini_llama::EnableCuda(model, device);
  else if (backend != "cpu") throw std::runtime_error("unknown backend: " + backend);
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
  if (!gguf_path.empty()) {
    term.PrintMessage("GGUF: " + gguf_path);
  }
  term.PrintMessage("Type /help for commands, /exit to quit.\n");
  //测试模型，如果上下文窗口小于等于256，打印提示信息
  if (model.config.max_seq_len <= 256) {
    term.PrintMessage(
        "Small model: generation is limited by its context window.");
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
      if (model.cuda) term.PrintMessage(mini_llama::CudaStats(model, ctx));
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
    if (tokens.empty()) {
      term.PrintMessage("Error: prompt produced no tokens.\n");
      continue;
    }

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

static int RunMain(int argc, char** argv) {
  InitTerminalUtf8();

  if (argc < 2 || std::strcmp(argv[1], "help") == 0 ||
      std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
    PrintUsage(argv[0]);
    return 0;
  }

  if (std::strcmp(argv[1], "--cuda-info") == 0) { std::cout << mini_llama::CudaDeviceInfo() << '\n'; return 0; }
  if (std::strcmp(argv[1], "bench") == 0) return RunGenerate(argc, argv, true);
  if (std::strcmp(argv[1], "generate") == 0) {//运行generate命令
    return RunGenerate(argc, argv);
  }
  if (std::strcmp(argv[1], "run") == 0) {
    return RunChat(argc, argv);
  }
  if (std::strcmp(argv[1], "inspect") == 0) {
    return RunInspect(argc, argv);
  }
  if (std::strcmp(argv[1], "inspect-gguf") == 0) {
    return RunInspectGguf(argc, argv);
  }
  if (std::strcmp(argv[1], "tokenize-gguf") == 0) {
    return RunTokenizeGguf(argc, argv);
  }

  std::cerr << "Unknown command: " << argv[1]
            << "\nExpected: generate, run, inspect, inspect-gguf or "
               "tokenize-gguf\n";
  PrintUsage(argv[0]);
  return 1;
}

int main(int argc, char** argv) {
  try { return RunMain(argc, argv); }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
