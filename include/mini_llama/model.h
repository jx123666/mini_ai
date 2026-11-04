//模型结构和配置：Transformer 的层数、维度、各层权重张量
#ifndef INCLUDE_MINI_LLAMA_MODEL_H_
#define INCLUDE_MINI_LLAMA_MODEL_H_

#include <string>
#include <vector>

#include "mini_llama/tensor.h"

namespace mini_llama {

enum class RopeType { kNormal, kNeoX };
struct CudaModelState;

struct ModelConfig {
  int vocab_size = 128;
  int dim = 32;
  int hidden_dim = 86;
  int n_layers = 2;
  int n_heads = 4;
  int n_kv_heads = 4;
  int head_dim = 8;//一般等于 dim / n_heads
  int max_seq_len = 128;
  float rope_theta = 10000.0f;
  float rms_norm_eps = 1e-5f;
  RopeType rope_type = RopeType::kNormal;
};

//一层 Transformer 的权重
struct LayerWeights {
  Tensor attention_norm;
  Tensor wq;
  Tensor wk;
  Tensor wv;
  // 没有偏置的模型保留空张量。
  Tensor bq;
  Tensor bk;
  Tensor bv;
  Tensor wo;
  Tensor ffn_norm;
  Tensor w_gate;
  Tensor w_up;
  Tensor w_down;
};

struct MiniLlamaModel {
  ModelConfig config;
  bool loaded = false;
  std::string load_error;
  std::shared_ptr<CudaModelState> cuda;

  Tensor token_embedding;
  std::vector<LayerWeights> layers;
  Tensor final_norm;
  Tensor lm_head;
};

//按 config 填全 0 权重，方便测形状链路
MiniLlamaModel MakeToyModel(const ModelConfig& config);

}

#endif
