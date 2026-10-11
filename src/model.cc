//按 config 造一个全 0 的玩具模型，给测试链路用
#include "mini_llama/model.h"

namespace mini_llama {
namespace {

//按 shape 填全 0
Tensor ZeroTensor(const std::vector<int>& shape) { return Tensor(shape, 0.0f); }

}  

MiniLlamaModel MakeToyModel(const ModelConfig& config) {
  MiniLlamaModel model;
  model.config = config;
  //head_dim没填时，按dim/n_heads推算
  if (model.config.head_dim <= 0 && model.config.n_heads > 0) {
    model.config.head_dim = model.config.dim / model.config.n_heads;
  }

  const ModelConfig& c = model.config;
  model.token_embedding = ZeroTensor({c.vocab_size, c.dim});
  model.layers.resize(c.n_layers);
  for (int i = 0; i < c.n_layers; ++i) {
    LayerWeights& lw = model.layers[i];
    lw.attention_norm = ZeroTensor({c.dim});
    //norm权重一般初始化成1，全0会让RmsNorm输出全0
    for (int d = 0; d < c.dim; ++d) {
      lw.attention_norm.data[d] = 1.0f;
    }
    //q,k,v,o权重
    lw.wq = ZeroTensor({c.n_heads * c.head_dim, c.dim});
    lw.wk = ZeroTensor({c.n_kv_heads * c.head_dim, c.dim});
    lw.wv = ZeroTensor({c.n_kv_heads * c.head_dim, c.dim});
    lw.wo = ZeroTensor({c.dim, c.n_heads * c.head_dim});
    lw.ffn_norm = ZeroTensor({c.dim});
    for (int d = 0; d < c.dim; ++d) {
      lw.ffn_norm.data[d] = 1.0f;
    }
    lw.w_gate = ZeroTensor({c.hidden_dim, c.dim});
    lw.w_up = ZeroTensor({c.hidden_dim, c.dim});
    lw.w_down = ZeroTensor({c.dim, c.hidden_dim});
  }
  //final_norm权重
  model.final_norm = ZeroTensor({c.dim});
  for (int d = 0; d < c.dim; ++d) {
    model.final_norm.data[d] = 1.0f;
  }
  //lm_head权重
  model.lm_head = ZeroTensor({c.vocab_size, c.dim});
  model.loaded = true;
  return model;
}

}
