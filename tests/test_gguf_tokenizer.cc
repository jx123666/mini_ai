//测从GGUF读出的BPE：roundtrip、special id、缺文件
#include <filesystem>
#include <string>
#include <vector>

#include "mini_llama/gguf_tokenizer.h"
#include "tests/gguf_tokenizer_fixture.h"
#include "tests/test_main.h"

using mini_llama::CreateGgufTokenizer;

//能打开这份小GGUF，词表长度和bos/eos要对上
static bool TestGgufTokenizerLoadsFromGguf() {
  std::filesystem::path path =
      mini_llama::test::WriteTinyGgufTokenizerFixture("tiny_tokenizer_load.gguf");
  auto tok = CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  MINI_LLAMA_ASSERT_EQ(tok->vocab_size(), 7);
  MINI_LLAMA_ASSERT_EQ(tok->bos_id(), 3);
  MINI_LLAMA_ASSERT_EQ(tok->eos_id(), 4);
  return true;
}

//"ab"按merges合成id=2，再解码回到原文
static bool TestGgufTokenizerEncodeDecodeRoundtrip() {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_tokenizer_roundtrip.gguf");
  auto tok = CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  const std::string text = "ab";
  const std::vector<int> ids = tok->Encode(text);
  MINI_LLAMA_ASSERT_EQ(ids.size(), 1u);
  MINI_LLAMA_ASSERT_EQ(ids[0], 2);
  MINI_LLAMA_ASSERT_TRUE(tok->Decode(ids) == text);
  return true;
}

//special整段一个id，eos解出来是</s>
static bool TestGgufTokenizerSpecialTokens() {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_tokenizer_special.gguf");
  auto tok = CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  const std::vector<int> start = tok->Encode("<|im_start|>");
  MINI_LLAMA_ASSERT_EQ(start.size(), 1u);
  MINI_LLAMA_ASSERT_EQ(start[0], 5);

  const std::vector<int> end = tok->Encode("<|im_end|>");
  MINI_LLAMA_ASSERT_EQ(end.size(), 1u);
  MINI_LLAMA_ASSERT_EQ(end[0], 6);

  MINI_LLAMA_ASSERT_EQ(tok->eos_id(), 4);
  MINI_LLAMA_ASSERT_TRUE(tok->DecodeToken(4) == "</s>");
  MINI_LLAMA_ASSERT_EQ(tok->bos_id(), 3);
  return true;
}

//本机有Qwen2时核对词表大小和几个特殊id
static bool TestGgufTokenizerLoadsRealModelWhenAvailable() {
  const std::string path = "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
  //没有这份大文件就跳过，不让干净仓库失败
  if (!std::filesystem::exists(path)) {
    return true;
  }
  auto tok = CreateGgufTokenizer(path);
  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  MINI_LLAMA_ASSERT_EQ(tok->vocab_size(), 151936);
  MINI_LLAMA_ASSERT_EQ(tok->bos_id(), 151643);
  MINI_LLAMA_ASSERT_EQ(tok->eos_id(), 151645);
  MINI_LLAMA_ASSERT_EQ(tok->Encode("<|im_start|>")[0], 151644);
  return true;
}

//文件不存在应返回空指针
static bool TestGgufTokenizerRejectsMissingFile() {
  auto tok = CreateGgufTokenizer("models/chat/nonexistent.gguf");
  MINI_LLAMA_ASSERT_TRUE(tok == nullptr);
  return true;
}

//启动时把这组测试登记进极简框架
static struct GgufTokenizerTestRegistrar {
  GgufTokenizerTestRegistrar() {
    RegisterTest("gguf_tokenizer_loads_from_gguf",
                 TestGgufTokenizerLoadsFromGguf);
    RegisterTest("gguf_tokenizer_encode_decode_roundtrip",
                 TestGgufTokenizerEncodeDecodeRoundtrip);
    RegisterTest("gguf_tokenizer_special_tokens",
                 TestGgufTokenizerSpecialTokens);
    RegisterTest("gguf_tokenizer_loads_real_model_when_available",
                 TestGgufTokenizerLoadsRealModelWhenAvailable);
    RegisterTest("gguf_tokenizer_rejects_missing_file",
                 TestGgufTokenizerRejectsMissingFile);
  }
} gguf_tokenizer_test_registrar;
