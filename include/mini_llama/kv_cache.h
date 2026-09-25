#ifndef INCLUDE_MINI_LLAMA_KV_CACHE_H_
#define INCLUDE_MINI_LLAMA_KV_CACHE_H_

#include "mini_llama/tensor.h"
namespace mini_llama {

//KV缓存，两块4D张量，一层一层、一个位置一个位置往里写
struct KvCache {
  Tensor keys;
  Tensor values;
  KvCache() = default;
  KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);
  //写入k，v
  void Write(int layer, int pos, const Tensor& k, const Tensor& v);
  //读取k，v
  const float* KeyPtr(int layer, int pos, int kv_head) const;
  const float* ValuePtr(int layer, int pos, int kv_head) const;
};

} 

#endif 
