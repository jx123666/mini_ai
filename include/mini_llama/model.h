//简单骨架，后面再加内容
#ifndef INCLUDE_MINI_LLAMA_MODEL_H_
#define INCLUDE_MINI_LLAMA_MODEL_H_

#include <string>

namespace mini_llama {

struct ModelConfig {
  int vocab_size = 128;
  int dim = 32;
  int hidden_dim = 86;
  int n_layers = 2;
  int n_heads = 4;
  int n_kv_heads = 4;
  int head_dim = 8;
  int max_seq_len = 128;
  float rope_theta = 10000.0f;
  float rms_norm_eps = 1e-5f;
};

//简单壳，后面再加embedding，layers
struct MiniLlamaModel {
  ModelConfig config;
  bool loaded = false;
  std::string load_error;
};

MiniLlamaModel MakeToyModel(const ModelConfig& config);

}  

#endif 
