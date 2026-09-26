#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/tokenizer.h"

using mini_llama::AsciiTokenizer;
using mini_llama::CreateBpeTokenizer;
using mini_llama::GgufTokenizer;

//去掉BOS之后"Hi!"能回来，整段解码会带上"<bos>"
TEST(TokenizerTest, AsciiRoundtrip) {
  AsciiTokenizer tok;
  const auto tokens = tok.Encode("Hi!");
  EXPECT_EQ(tokens[0], tok.bos_id());
  const std::vector<int> payload(tokens.begin() + 1, tokens.end());
  EXPECT_EQ(tok.Decode(payload), "Hi!");
  EXPECT_EQ(tok.Decode(tokens), "<bos>Hi!");
}

//词表里"hello"的编号是7，合并完应该只剩这一个id
TEST(TokenizerTest, BpeHelloBecomesOneToken) {
  const auto dir = std::filesystem::temp_directory_path() / "mini_ai_gtest_bpe";
  std::filesystem::create_directories(dir);
  const auto vocab = dir / "vocab.json";
  const auto merges = dir / "merges.txt";
  const auto special = dir / "special.json";
  {
    std::ofstream out(vocab);
    out << "{\"h\":0,\"e\":1,\"l\":2,\"o\":3,\"he\":4,\"hel\":5,\"hell\":6,"
           "\"hello\":7,\"<unk>\":9}";
  }
  {
    std::ofstream out(merges);
    out << "h e\nhe l\nhel l\nhell o\n";
  }
  {
    std::ofstream out(special);
    out << "{\"bos_id\":1,\"eos_id\":2,\"unk_id\":9}";
  }

  auto tok = CreateBpeTokenizer(vocab.string(), merges.string(),
                                special.string());
  ASSERT_TRUE(tok != nullptr);
  const auto ids = tok->Encode("hello");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 7);
  EXPECT_EQ(tok->Decode(ids), "hello");
  std::filesystem::remove_all(dir);
}

//和BPE同一套合并，只是词表直接塞进去，结果还是7
TEST(TokenizerTest, GgufPiecesSameMerge) {
  GgufTokenizer tok;
  ASSERT_TRUE(tok.LoadPieces(
      {"h", "e", "l", "o", "he", "hel", "hell", "hello"},
      {"h e", "he l", "hel l", "hell o"}, 1, 2, 0));
  const auto ids = tok.Encode("hello");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 7);
  EXPECT_EQ(tok.Decode(ids), "hello");
}
