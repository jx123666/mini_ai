#include <stdexcept>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "tests/test_main.h"

using mini_llama::ForwardBatch;
using mini_llama::ForwardToken;
using mini_llama::LoadModel;
using mini_llama::MakeToyModel;
using mini_llama::MiniBatch;
using mini_llama::MiniLlamaContext;
using mini_llama::MiniLlamaModel;
using mini_llama::ModelConfig;
using mini_llama::Tensor;

//Forward 用例优先加载 tiny 真权重；失败再退回零权重玩具模型
static MiniLlamaModel MakeSmallToy() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (model.loaded && !model.layers.empty()) {
    return model;
  }
  ModelConfig cfg;
  cfg.vocab_size = 128;
  cfg.dim = 8;
  cfg.hidden_dim = 16;
  cfg.n_layers = 2;
  cfg.n_heads = 2;
  cfg.n_kv_heads = 2;
  cfg.head_dim = 4;
  cfg.max_seq_len = 32;
  return MakeToyModel(cfg);
}

//一个 token、一个位置
static bool TestBatchSingle() {
  MiniBatch b = MiniBatch::Single(42, 5);
  MINI_LLAMA_ASSERT_EQ(b.num_tokens(), 1);
  MINI_LLAMA_ASSERT_EQ(b.tokens[0], 42);
  MINI_LLAMA_ASSERT_EQ(b.positions[0], 5);
  return true;
}

//位置从0连续往上加
static bool TestBatchFromTokens() {
  std::vector<int> toks = {10, 20, 30};
  MiniBatch b = MiniBatch::FromTokens(toks, 0);
  MINI_LLAMA_ASSERT_EQ(b.num_tokens(), 3);
  MINI_LLAMA_ASSERT_EQ(b.tokens[0], 10);
  MINI_LLAMA_ASSERT_EQ(b.tokens[1], 20);
  MINI_LLAMA_ASSERT_EQ(b.tokens[2], 30);
  MINI_LLAMA_ASSERT_EQ(b.positions[0], 0);
  MINI_LLAMA_ASSERT_EQ(b.positions[1], 1);
  MINI_LLAMA_ASSERT_EQ(b.positions[2], 2);
  return true;
}

//FromTokens带起始位置偏移
static bool TestBatchFromTokensWithOffset() {
  std::vector<int> toks = {5, 6};
  MiniBatch b = MiniBatch::FromTokens(toks, 10);
  MINI_LLAMA_ASSERT_EQ(b.positions[0], 10);
  MINI_LLAMA_ASSERT_EQ(b.positions[1], 11);
  return true;
}

//空batch的 num_tokens应该是 0
static bool TestBatchEmpty() {
  MiniBatch b;
  MINI_LLAMA_ASSERT_EQ(b.num_tokens(), 0);
  return true;
}

//batch跑一遍应该和逐个ForwardToken结果一样
static bool TestForwardBatchPrefillMatchesIndividual() {
  MiniLlamaModel model = MakeSmallToy();
  std::vector<int> tokens = {1, 10, 11, 12};

  MiniLlamaContext ctx1(&model);
  Tensor logits1;
  for (size_t i = 0; i < tokens.size(); ++i) {
    ctx1.pos = static_cast<int>(i);
    logits1 = ForwardToken(ctx1, model, tokens[i]);
  }

  MiniLlamaContext ctx2(&model);
  MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
  Tensor logits2 = ForwardBatch(ctx2, model, batch);

  MINI_LLAMA_ASSERT_EQ(logits1.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(logits2.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(logits1.shape[0], logits2.shape[0]);
  for (size_t i = 0; i < logits1.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(logits1.data[i], logits2.data[i], 1e-6f);
  }
  return true;
}

//更新 context；用 tiny 真权重，才能看到 cache 里写出非零
static bool TestForwardBatchUpdatesContext() {
  MiniLlamaModel model = MakeSmallToy();
  if (!model.loaded || model.layers.empty()) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load tiny model");
  }
  std::vector<int> tokens = {1, 10, 11};
  MiniLlamaContext ctx(&model);
  MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
  ForwardBatch(ctx, model, batch);

  MINI_LLAMA_ASSERT_EQ(ctx.token_history.size(), tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    MINI_LLAMA_ASSERT_EQ(ctx.token_history[i], tokens[i]);
  }
  MINI_LLAMA_ASSERT_EQ(ctx.pos, static_cast<int>(tokens.size()) - 1);
  MINI_LLAMA_ASSERT_EQ(ctx.n_prefill_tokens, 3);
  MINI_LLAMA_ASSERT_EQ(ctx.n_decode_tokens, 0);

  //确认 pos=1 的 key 已经写进去了
  const float* k = ctx.kv_cache.KeyPtr(0, 1, 0);
  MINI_LLAMA_ASSERT_TRUE(k[0] != 0.0f);
  return true;
}

//先Prefill再Decode，历史继续往后面追加，decode计数+1
static bool TestForwardBatchAppendsDecodeHistory() {
  MiniLlamaModel model = MakeSmallToy();
  MiniLlamaContext ctx(&model);

  MiniBatch prefill = MiniBatch::FromTokens({1, 10}, 0);
  ForwardBatch(ctx, model, prefill);

  MiniBatch decode = MiniBatch::Single(11, 2);
  ForwardBatch(ctx, model, decode);

  MINI_LLAMA_ASSERT_EQ(ctx.token_history.size(), 3);
  MINI_LLAMA_ASSERT_EQ(ctx.token_history[0], 1);
  MINI_LLAMA_ASSERT_EQ(ctx.token_history[1], 10);
  MINI_LLAMA_ASSERT_EQ(ctx.token_history[2], 11);
  MINI_LLAMA_ASSERT_EQ(ctx.pos, 2);
  MINI_LLAMA_ASSERT_EQ(ctx.n_prefill_tokens, 2);
  MINI_LLAMA_ASSERT_EQ(ctx.n_decode_tokens, 1);
  return true;
}

//空batch报错
static bool TestForwardBatchEmptyRejected() {
  MiniLlamaModel model = MakeSmallToy();
  MiniLlamaContext ctx(&model);
  MiniBatch batch;
  try {
    ForwardBatch(ctx, model, batch);
    MINI_LLAMA_ASSERT_FAIL("expected exception for empty batch");
  } catch (const std::runtime_error&) {
  }
  return true;
}

//tokens和positions长度不对报错
static bool TestForwardBatchMismatchedSizesRejected() {
  MiniLlamaModel model = MakeSmallToy();
  MiniLlamaContext ctx(&model);
  MiniBatch batch;
  batch.tokens = {1, 2};
  batch.positions = {0};
  try {
    ForwardBatch(ctx, model, batch);
    MINI_LLAMA_ASSERT_FAIL("expected exception for size mismatch");
  } catch (const std::runtime_error&) {
  }
  return true;
}

static struct BatchTestRegistrar {
  BatchTestRegistrar() {
    RegisterTest("batch_single", TestBatchSingle);
    RegisterTest("batch_from_tokens", TestBatchFromTokens);
    RegisterTest("batch_from_tokens_with_offset",
                 TestBatchFromTokensWithOffset);
    RegisterTest("batch_empty", TestBatchEmpty);
    RegisterTest("forward_batch_prefill_matches_individual",
                 TestForwardBatchPrefillMatchesIndividual);
    RegisterTest("forward_batch_updates_context",
                 TestForwardBatchUpdatesContext);
    RegisterTest("forward_batch_appends_decode_history",
                 TestForwardBatchAppendsDecodeHistory);
    RegisterTest("forward_batch_empty_rejected", TestForwardBatchEmptyRejected);
    RegisterTest("forward_batch_mismatched_sizes_rejected",
                 TestForwardBatchMismatchedSizesRejected);
  }
} batch_test_registrar;
