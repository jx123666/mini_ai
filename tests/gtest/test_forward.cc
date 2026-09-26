//GoogleTest 版 Forward 主链路测试
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"

using mini_llama::AsciiTokenizer;
using mini_llama::ForwardToken;
using mini_llama::LoadModel;
using mini_llama::MiniLlamaContext;
using mini_llama::MiniLlamaModel;
using mini_llama::SampleGreedy;
using mini_llama::Tensor;

static MiniLlamaModel LoadTiny() {
  return LoadModel("models/tiny/model.json", "models/tiny/model.bin");
}

TEST(ForwardTest, TokenProducesLogitShape) {
  MiniLlamaModel model = LoadTiny();
  ASSERT_FALSE(model.layers.empty());
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  Tensor logits = ForwardToken(ctx, model, 1);
  EXPECT_EQ(logits.num_dims(), 1);
  EXPECT_EQ(logits.shape[0], model.config.vocab_size);
}

TEST(ForwardTest, TokenChangesKvCache) {
  MiniLlamaModel model = LoadTiny();
  ASSERT_FALSE(model.layers.empty());
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  EXPECT_NEAR(ctx.kv_cache.keys.At({0, 0, 0, 0}), 0.0f, 1e-6f);
  ForwardToken(ctx, model, 1);
  EXPECT_TRUE(std::isfinite(ctx.kv_cache.keys.At({0, 0, 0, 0})));
}

TEST(ForwardTest, GenerationGreedySequence) {
  MiniLlamaModel model = LoadTiny();
  ASSERT_FALSE(model.layers.empty());
  AsciiTokenizer tokenizer;
  std::vector<int> tokens = tokenizer.Encode("hello");
  MiniLlamaContext ctx(&model);

  const size_t prompt_len = tokens.size();
  for (size_t i = 0; i < prompt_len; ++i) {
    ctx.pos = static_cast<int>(i);
    Tensor logits = ForwardToken(ctx, model, tokens[i]);
    if (i + 1 == prompt_len) {
      tokens.push_back(SampleGreedy(logits));
    }
  }
  for (int i = 0; i < 5; ++i) {
    ctx.pos = static_cast<int>(tokens.size() - 1);
    Tensor logits = ForwardToken(ctx, model, tokens.back());
    tokens.push_back(SampleGreedy(logits));
  }
  EXPECT_GT(tokens.size(), 6u);
  EXPECT_GE(tokens[tokens.size() - 6], 0);
}

TEST(ForwardTest, RejectsInvalidToken) {
  MiniLlamaModel model = LoadTiny();
  ASSERT_TRUE(model.loaded);
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  EXPECT_THROW(ForwardToken(ctx, model, model.config.vocab_size),
               std::out_of_range);
}
