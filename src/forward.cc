//Forward主链路 embed -> rmsnorm -> qkv -> rope -> attention -> ffn -> logits
#include "mini_llama/forward.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/ops.h"
#include "mini_llama/backend.h"

namespace mini_llama {
namespace {
//验证输入
void ValidateForwardInputs(const MiniLlamaContext& ctx,
                           const MiniLlamaModel& model, int token) {
  const ModelConfig& c = model.config;
  if (!model.loaded) {
    throw std::runtime_error("ForwardToken called with an unloaded model");
  }
  if (c.n_heads <= 0 || c.n_kv_heads <= 0 || c.head_dim <= 0 ||
      c.n_heads % c.n_kv_heads != 0 || c.head_dim % 2 != 0 ||
      c.dim != c.n_heads * c.head_dim) {
    throw std::runtime_error("ForwardToken invalid attention dimensions");
  }
  if (token < 0 || token >= c.vocab_size) {
    throw std::out_of_range("ForwardToken token id out of range");
  }
  if (ctx.pos < 0 || ctx.pos >= c.max_seq_len) {
    throw std::out_of_range("ForwardToken position out of range");
  }
  if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
    throw std::runtime_error("ForwardToken layer count mismatch");
  }
  //验证token_embedding是否足够
  if (model.token_embedding.data.size() <
      static_cast<size_t>(c.vocab_size) * static_cast<size_t>(c.dim)) {
    throw std::runtime_error("ForwardToken embedding too small");
  }
}

//嵌入token
Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
  const int dim = model.config.dim;
  Tensor x({dim}, 0.0f);
  for (int i = 0; i < dim; ++i) {
    x.data[i] = model.token_embedding.data[token_id * dim + i];
  }
  return x;
}

//GQA：几个 query head共用一个kv head
int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
  return q_head / (n_heads / n_kv_heads);
}

Tensor ProjectQkv(const Tensor& x, const Tensor& weight, const Tensor& bias) {
  Tensor y = Linear(x, weight);
  if (!bias.data.empty()) {
    bias.AssertShape(y.shape, "QKV bias");
    y = ElementwiseAdd(y, bias);
  }
  return y;
}

//attention:写当前K/V进 cache，再拿历史算加权和
Tensor AttentionForward(const Tensor& q, const Tensor& k, const Tensor& v,
                        int pos, int layer, KvCache& kv_cache, int n_heads,
                        int n_kv_heads, int head_dim) {
  kv_cache.Write(layer, pos, k, v);
  //输出
  Tensor attn_out({n_heads, head_dim}, 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  //遍历head
  for (int h = 0; h < n_heads; ++h) {
    const int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);

    //当前query和0..pos每个位置的key算分数
    Tensor scores({pos + 1}, 0.0f);
    for (int t = 0; t <= pos; ++t) {
      //获取key
      const float* k_ptr = kv_cache.KeyPtr(layer, t, kv_head);
      //计算分数
      float dot = 0.0f;
      //遍历head_dim
      for (int d = 0; d < head_dim; ++d) {
        dot += q.data[h * head_dim + d] * k_ptr[d];
      }
      scores.data[t] = dot * scale;
    }
    Tensor probs = Softmax(scores);

    //用概率去加权历史value
    for (int d = 0; d < head_dim; ++d) {
      float val = 0.0f;
      for (int t = 0; t <= pos; ++t) {
        const float* v_ptr = kv_cache.ValuePtr(layer, t, kv_head);
        val += probs.data[t] * v_ptr[d];
      }
      attn_out.data[h * head_dim + d] = val;
    }
  }
  return attn_out;
}

//SwiGLU，gate/up投影后再 down投影回dim
Tensor FfnForward(const Tensor& h, const LayerWeights& lw) {
  Tensor gate = Linear(h, lw.w_gate);
  Tensor up = Linear(h, lw.w_up);
  return Linear(SwiGlu(gate, up), lw.w_down);
}

//单层forward
Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const Tensor& x, int layer, const LayerWeights& lw) {
  const ModelConfig& c = model.config;
  const int dim = c.dim;
  const int n_heads = c.n_heads;
  const int n_kv_heads = c.n_kv_heads;
  const int head_dim = c.head_dim;
  const int pos = ctx.pos;

  //attention 子层
  Tensor h = RmsNorm(x, lw.attention_norm, c.rms_norm_eps);

  //qkv投影，再拆成多头
  Tensor q = ProjectQkv(h, lw.wq, lw.bq).ReshapeChecked({n_heads, head_dim}, "q");
  Tensor k = ProjectQkv(h, lw.wk, lw.bk).ReshapeChecked({n_kv_heads, head_dim}, "k");
  Tensor v = ProjectQkv(h, lw.wv, lw.bv).ReshapeChecked({n_kv_heads, head_dim}, "v");

  Rope(q, k, pos, c.rope_theta, c.rope_type);

  Tensor attn_out =
      AttentionForward(q, k, v, pos, layer, ctx.kv_cache, n_heads, n_kv_heads,
                       head_dim);
  //多头拼回一维再投影
  Tensor attn_flat =
      attn_out.ReshapeChecked({n_heads * head_dim}, "attn_flat");
  Tensor attn_proj = Linear(attn_flat, lw.wo);
  Tensor x_attn = ElementwiseAdd(x, attn_proj);

  //ffn 子层 
  Tensor h2 = RmsNorm(x_attn, lw.ffn_norm, c.rms_norm_eps);
  Tensor ff = FfnForward(h2, lw);
  return ElementwiseAdd(x_attn, ff);
}

//最后再norm一次，经lm_head得到词表分数
Tensor ComputeLogits(const Tensor& x, const MiniLlamaModel& model) {
  const ModelConfig& c = model.config;
  Tensor normed = RmsNorm(x, model.final_norm, c.rms_norm_eps);
  return Linear(normed, model.lm_head)
      .ReshapeChecked({c.vocab_size}, "logits");
}

} 

Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token) {
  ValidateForwardInputs(ctx, model, token);
  if (model.cuda) return ForwardCuda(ctx, model, token);

  //embed
  Tensor x = EmbedToken(model, token);

  //一层一层往下走
  for (int layer = 0; layer < model.config.n_layers; ++layer) {
    x = ForwardLayer(ctx, model, x, layer, model.layers[layer]);
  }

  //计算logits
  return ComputeLogits(x, model);
}

//批量forward
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch) {
  if (batch.num_tokens() == 0) {
    throw std::runtime_error("ForwardBatch: empty batch");
  }
  if (batch.tokens.size() != batch.positions.size()) {
    throw std::runtime_error(
        "ForwardBatch: tokens and positions size mismatch");
  }
//是否作为prefill
  const bool treat_as_prefill =
      ctx.token_history.empty() || batch.num_tokens() > 1;
  //遍历batch
  Tensor logits;
  for (int i = 0; i < batch.num_tokens(); ++i) {
    //设置位置
    ctx.pos = batch.positions[i];
    //forward token
    logits = ForwardToken(ctx, model, batch.tokens[i]);
    //添加到token历史
    ctx.token_history.push_back(batch.tokens[i]);
    if (treat_as_prefill) {
      ++ctx.n_prefill_tokens;
    } else {
      ++ctx.n_decode_tokens;
    }
  }
  return logits;
}

}
