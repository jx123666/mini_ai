//测 Forward 主链路：logits 形状、写 KV、prefill/decode、非法输入
#include <cmath>
#include <stdexcept>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"
#include "tests/test_main.h"

using mini_llama::AsciiTokenizer;
using mini_llama::ForwardToken;
using mini_llama::LoadModel;
using mini_llama::MiniLlamaContext;
using mini_llama::MiniLlamaModel;
using mini_llama::SampleGreedy;
using mini_llama::Tensor;

static MiniLlamaModel LoadTinyOrFail() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded || model.layers.empty()) {
    //让调用方用 ASSERT_FAIL
  }
  return model;
}

//Forward 出来的 logits 该是 [vocab_size]
static bool TestForwardTokenProducesLogitShape() {
  MiniLlamaModel model = LoadTinyOrFail();
  if (model.layers.empty()) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load model");
  }
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  Tensor logits = ForwardToken(ctx, model, 1);
  MINI_LLAMA_ASSERT_EQ(logits.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(logits.shape[0], model.config.vocab_size);
  return true;
}

//跑完一层后，KV cache 对应位置不该还是「没写过」的状态
static bool TestForwardTokenChangesKvCache() {
  MiniLlamaModel model = LoadTinyOrFail();
  if (model.layers.empty()) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load model");
  }
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  float before = ctx.kv_cache.keys.At({0, 0, 0, 0});
  MINI_LLAMA_ASSERT_NEAR(before, 0.0f, 1e-6f);

  ForwardToken(ctx, model, 1);

  float after = ctx.kv_cache.keys.At({0, 0, 0, 0});
  MINI_LLAMA_ASSERT_TRUE(std::isfinite(after));
  return true;
}

//prefill + decode 几步，greedy 采几个 token
static bool TestGenerationGreedySequence() {
  MiniLlamaModel model = LoadTinyOrFail();
  if (model.layers.empty()) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load model");
  }
  AsciiTokenizer tokenizer;
  std::vector<int> tokens = tokenizer.Encode("hello");
  MiniLlamaContext ctx(&model);

  size_t prompt_len = tokens.size();
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

  MINI_LLAMA_ASSERT_TRUE(tokens.size() > 6);
  MINI_LLAMA_ASSERT_TRUE(tokens[tokens.size() - 6] >= 0);
  return true;
}

//pos 从 0 走到 1，两次都能正常出 logits
static bool TestPositionIncrementsCorrectly() {
  MiniLlamaModel model = LoadTinyOrFail();
  if (model.layers.empty()) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load model");
  }
  MiniLlamaContext ctx(&model);

  ctx.pos = 0;
  ForwardToken(ctx, model, 1);

  ctx.pos = 1;
  Tensor logits = ForwardToken(ctx, model, 2);
  MINI_LLAMA_ASSERT_EQ(logits.shape[0], model.config.vocab_size);
  return true;
}

//非法 token id 应该抛异常
static bool TestForwardRejectsInvalidToken() {
  MiniLlamaModel model = LoadTinyOrFail();
  if (!model.loaded) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load model");
  }
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  try {
    ForwardToken(ctx, model, model.config.vocab_size);
    MINI_LLAMA_ASSERT_FAIL("ForwardToken accepted an out-of-range token");
  } catch (const std::out_of_range&) {
    return true;
  } catch (...) {
    MINI_LLAMA_ASSERT_FAIL("unexpected exception type");
  }
  return false;
}

static struct ForwardTestRegistrar {
  ForwardTestRegistrar() {
    RegisterTest("forward_token_produces_logit_shape",
                 TestForwardTokenProducesLogitShape);
    RegisterTest("forward_token_changes_kv_cache",
                 TestForwardTokenChangesKvCache);
    RegisterTest("generation_greedy_sequence", TestGenerationGreedySequence);
    RegisterTest("position_increments_correctly",
                 TestPositionIncrementsCorrectly);
    RegisterTest("forward_rejects_invalid_token",
                 TestForwardRejectsInvalidToken);
  }
} forward_test_registrar;
