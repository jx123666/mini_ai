#include "mini_llama/forward.h"
#include <stdexcept>
#include <string>

namespace mini_llama {
namespace {

//检查模型已加载、token和位置不越界
void ValidateForwardInputs(const MiniLlamaContext& ctx,
                           const MiniLlamaModel& model, int token) {
  const ModelConfig& c = model.config;
  if (!model.loaded) {
    throw std::runtime_error("ForwardToken called with an unloaded model");
  }
  if (token < 0 || token >= c.vocab_size) {
    throw std::out_of_range("ForwardToken token id out of range");
  }
  if (ctx.pos < 0 || ctx.pos >= c.max_seq_len) {
    throw std::out_of_range("ForwardToken position out of range");
  }
}

Tensor MakeToyKv(int n_kv_heads, int head_dim, int token, int pos, int layer,
                 float base) {
  Tensor t({n_kv_heads, head_dim}, 0.0f);
  for (int h = 0; h < n_kv_heads; ++h) {
    for (int d = 0; d < head_dim; ++d) {
      //值跟token，pos，layer，head有关系
      t.At2(h, d) = base + static_cast<float>(token) * 0.01f +
                    static_cast<float>(pos) + static_cast<float>(layer) * 10.0f +
                    static_cast<float>(h) * 0.1f + static_cast<float>(d) * 0.001f;
    }
  }
  return t;
}

}

//处理1个token，返回logits
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token) {
  ValidateForwardInputs(ctx, model, token);
  const ModelConfig& c = model.config;

  //每一层都往cache里写一份当前位置的 k、v
  for (int layer = 0; layer < c.n_layers; ++layer) {
    Tensor k = MakeToyKv(c.n_kv_heads, c.head_dim, token, ctx.pos, layer, 1.0f);
    Tensor v =
        MakeToyKv(c.n_kv_heads, c.head_dim, token, ctx.pos, layer, 100.0f);
    ctx.kv_cache.Write(layer, ctx.pos, k, v);
  }
  Tensor logits({c.vocab_size}, 0.0f);
  for (int i = 0; i < c.vocab_size; ++i) {
    logits.data[i] = static_cast<float>(i + ctx.pos) * 0.001f;
  }
  if (token >= 0 && token < c.vocab_size) {
    logits.data[token] += 1.0f;
  }
  return logits;
}

//一批 token顺序处理，返回最后一个的logits
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch) {
  //空batch或tokens/positions长度对不上，异常
  if (batch.num_tokens() == 0) {
    throw std::runtime_error("ForwardBatch: empty batch");
  }
  if (batch.tokens.size() != batch.positions.size()) {
    throw std::runtime_error(
        "ForwardBatch: tokens and positions size mismatch");
  }
  const bool treat_as_prefill =
      ctx.token_history.empty() || batch.num_tokens() > 1;

  Tensor logits;
  for (int i = 0; i < batch.num_tokens(); ++i) {
    ctx.pos = batch.positions[i];//摆到这个 token 该在的位置
    logits = ForwardToken(ctx, model, batch.tokens[i]);
    ctx.token_history.push_back(batch.tokens[i]);
    if (treat_as_prefill) {
      ++ctx.n_prefill_tokens;
    } else {
      ++ctx.n_decode_tokens;
    }
  }
  return logits;//返回最后一个token的分数
}

}
