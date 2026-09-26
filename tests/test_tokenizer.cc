//测分词：ASCII能编能解，JSON最长匹配，BPE把hello收成一个id再解回去
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/tokenizer.h"
#include "tests/test_main.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using mini_llama::AsciiTokenizer;
using mini_llama::BpeTokenizer;
using mini_llama::CreateBpeTokenizer;
using mini_llama::CreateTokenizer;
using mini_llama::GgufTokenizer;
using mini_llama::JsonVocabTokenizer;

namespace {

//临时目录，测试结束自动删掉，避免词表文件留在磁盘上
struct TempDir {
  std::filesystem::path dir;
  explicit TempDir(const std::string& name) {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    dir = std::filesystem::temp_directory_path() /
          (name + "_" + std::to_string(stamp));
    std::filesystem::create_directories(dir);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

//引号和反斜杠写进JSON时要加反斜杠，不然词表解析会断掉
std::string JsonEscape(char c) {
  if (c == '"' || c == '\\') {
    return std::string("\\") + c;
  }
  return std::string(1, c);
}

//写出128个ASCII词，id就是字符编号，控制字符内容为空
void WriteAsciiVocab(const std::filesystem::path& path) {
  std::ofstream out(path);
  out << "[\n";
  out << "{\"id\":0,\"content\":\"<unk>\",\"special\":true},\n";
  out << "{\"id\":1,\"content\":\"<bos>\",\"special\":true},\n";
  out << "{\"id\":2,\"content\":\"<eos>\",\"special\":true}";
  for (int i = 3; i < 128; ++i) {
    out << ",\n{\"id\":" << i << ",\"content\":\"";
    if (i >= 32 && i != 127) {
      out << JsonEscape(static_cast<char>(i));
    }
    out << "\",\"special\":false}";
  }
  out << "\n]\n";
}

//写出hello的词表和合并规则，"hello"的编号事先写成7
void WriteHelloBpe(const std::filesystem::path& dir,
                   std::filesystem::path* vocab, std::filesystem::path* merges,
                   std::filesystem::path* special) {
  *vocab = dir / "vocab.json";
  *merges = dir / "merges.txt";
  *special = dir / "special_tokens.json";
  {
    std::ofstream out(*vocab);
    out << "{\n"
        << "  \"h\": 0,\n"
        << "  \"e\": 1,\n"
        << "  \"l\": 2,\n"
        << "  \"o\": 3,\n"
        << "  \"he\": 4,\n"
        << "  \"hel\": 5,\n"
        << "  \"hell\": 6,\n"
        << "  \"hello\": 7,\n"
        << "  \"<|im_start|>\": 8,\n"
        << "  \"<unk>\": 9\n"
        << "}\n";
  }
  {
    std::ofstream out(*merges);
    out << "h e\n"
        << "he l\n"
        << "hel l\n"
        << "hell o\n";
  }
  {
    std::ofstream out(*special);
    out << "{ \"bos_id\": 1, \"eos_id\": 2, \"unk_id\": 9 }\n";
  }
}

//Encode("Hi!") = [BOS, 'H', 'i', '!']，去掉BOS后解码才是原文
bool TestAsciiEncodeDecodeRoundtrip() {
  AsciiTokenizer tok;
  const std::string text = "Hi!";
  const auto tokens = tok.Encode(text);
  MINI_LLAMA_ASSERT_EQ(tokens.size(), 4);
  MINI_LLAMA_ASSERT_EQ(tokens[0], tok.bos_id());
  MINI_LLAMA_ASSERT_EQ(tokens[1], static_cast<int>('H'));
  MINI_LLAMA_ASSERT_EQ(tokens[2], static_cast<int>('i'));
  MINI_LLAMA_ASSERT_EQ(tokens[3], static_cast<int>('!'));

  //完整解码带<bos>；去掉第一个id后，可见字符回到原文
  MINI_LLAMA_ASSERT_TRUE(tok.Decode(tokens) == "<bos>Hi!");
  const std::vector<int> payload(tokens.begin() + 1, tokens.end());
  MINI_LLAMA_ASSERT_TRUE(tok.Decode(payload) == text);
  return true;
}

//超过127的字节词表里没有，应该记成UNK
bool TestAsciiUnknownByte() {
  AsciiTokenizer tok;
  std::string text;
  text.push_back(static_cast<char>(200));
  const auto tokens = tok.Encode(text);
  MINI_LLAMA_ASSERT_EQ(tokens.size(), 2);
  MINI_LLAMA_ASSERT_EQ(tokens[1], tok.unk_id());
  return true;
}

//128个ASCII词时，JSON编码应和ASCII一样；有"hello"时整段吃掉
bool TestJsonVocabMatchesAsciiAndLongestMatch() {
  TempDir tmp("mini_ai_json_vocab");
  const auto vocab = tmp.dir / "vocab.json";
  WriteAsciiVocab(vocab);

  JsonVocabTokenizer tok(vocab.string());
  MINI_LLAMA_ASSERT_EQ(tok.vocab_size(), 128);
  AsciiTokenizer ascii;
  const auto a = ascii.Encode("hi");
  const auto j = tok.Encode("hi");
  MINI_LLAMA_ASSERT_EQ(a.size(), j.size());
  for (size_t i = 0; i < a.size(); ++i) {
    MINI_LLAMA_ASSERT_EQ(a[i], j[i]);
  }
  MINI_LLAMA_ASSERT_TRUE(tok.Decode(j) == "<bos>hi");

  //另写一份带多字符词的词表，确认最长匹配先吃"hello"
  const auto long_vocab = tmp.dir / "long.json";
  {
    std::ofstream out(long_vocab);
    out << "["
        << "{\"id\":0,\"content\":\"<unk>\",\"special\":true},"
        << "{\"id\":1,\"content\":\"<bos>\",\"special\":true},"
        << "{\"id\":2,\"content\":\"h\",\"special\":false},"
        << "{\"id\":3,\"content\":\"hello\",\"special\":false}"
        << "]";
  }
  JsonVocabTokenizer long_tok(long_vocab.string());
  const auto hello = long_tok.Encode("hello");
  MINI_LLAMA_ASSERT_EQ(hello.size(), 2);
  MINI_LLAMA_ASSERT_EQ(hello[0], 1);
  MINI_LLAMA_ASSERT_EQ(hello[1], 3);
  return true;
}

//没给词表路径时，工厂退回ASCII，词表大小仍是128
bool TestFactoryFallsBackToAscii() {
  auto tok = CreateTokenizer("");
  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  MINI_LLAMA_ASSERT_EQ(tok->vocab_size(), 128);
  return true;
}

//合并完只剩"hello"，词表里它的编号是7，再解码回到"hello"
bool TestBpeMergeAndRoundtrip() {
  TempDir tmp("mini_ai_bpe");
  std::filesystem::path vocab;
  std::filesystem::path merges;
  std::filesystem::path special;
  WriteHelloBpe(tmp.dir, &vocab, &merges, &special);

  auto tok = CreateBpeTokenizer(vocab.string(), merges.string(),
                                special.string());
  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);

  const auto hello = tok->Encode("hello");
  MINI_LLAMA_ASSERT_EQ(hello.size(), 1u);
  MINI_LLAMA_ASSERT_EQ(hello[0], 7);
  MINI_LLAMA_ASSERT_TRUE(tok->Decode(hello) == "hello");

  const auto special_ids = tok->Encode("<|im_start|>");
  MINI_LLAMA_ASSERT_EQ(special_ids.size(), 1u);
  MINI_LLAMA_ASSERT_EQ(special_ids[0], 8);
  MINI_LLAMA_ASSERT_TRUE(tok->Decode(special_ids) == "<|im_start|>");
  return true;
}

//GGUF这边只是换了喂词表的方式，合并结果同样是编号7
bool TestGgufUsesSameMerge() {
  GgufTokenizer tok;
  const bool ok = tok.LoadPieces(
      {"h", "e", "l", "o", "he", "hel", "hell", "hello"},
      {"h e", "he l", "hel l", "hell o"}, 1, 2, 0);
  MINI_LLAMA_ASSERT_TRUE(ok);
  const auto ids = tok.Encode("hello");
  MINI_LLAMA_ASSERT_EQ(ids.size(), 1u);
  MINI_LLAMA_ASSERT_EQ(ids[0], 7);
  MINI_LLAMA_ASSERT_TRUE(tok.Decode(ids) == "hello");
  return true;
}

//启动时把分词测试登记进极简框架
struct TokenizerTestRegistrar {
  TokenizerTestRegistrar() {
    RegisterTest("ascii_encode_decode_roundtrip", TestAsciiEncodeDecodeRoundtrip);
    RegisterTest("ascii_unknown_byte", TestAsciiUnknownByte);
    RegisterTest("json_vocab_matches_ascii_and_longest_match",
                 TestJsonVocabMatchesAsciiAndLongestMatch);
    RegisterTest("factory_falls_back_to_ascii", TestFactoryFallsBackToAscii);
    RegisterTest("bpe_merge_and_roundtrip", TestBpeMergeAndRoundtrip);
    RegisterTest("gguf_uses_same_merge", TestGgufUsesSameMerge);
  }
} g_tokenizer_test_registrar;

}
