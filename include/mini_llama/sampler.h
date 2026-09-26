//greedy直接取logits最大值下标
#ifndef INCLUDE_MINI_LLAMA_SAMPLER_H_
#define INCLUDE_MINI_LLAMA_SAMPLER_H_

#include "mini_llama/ops.h"
#include "mini_llama/tensor.h"

namespace mini_llama {
//采样参数
struct SamplingParams {
  float temperature = 1.0f;
  int top_k = 0;
  //随机种子
  int seed = 0;
};
//greedy采样
inline int SampleGreedy(const Tensor& logits) { return ArgMax(logits); }

}

#endif
