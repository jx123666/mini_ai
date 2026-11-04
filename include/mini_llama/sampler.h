//采样：从 logits 里挑下一个 token
//greedy 取最大；temperature 按概率抽；top-k 只在前 k 个里抽
#ifndef INCLUDE_MINI_LLAMA_SAMPLER_H_
#define INCLUDE_MINI_LLAMA_SAMPLER_H_

#include <random>

#include "mini_llama/tensor.h"

namespace mini_llama {

struct SamplingParams {
  float temperature = 0.0f;//0 表示 greedy
  int top_k = 0;//0 表示不启用 top-k
  unsigned int seed = 0;//0 表示用随机设备
};

class MiniSampler {
 public:
  explicit MiniSampler(unsigned int seed = 0);
  explicit MiniSampler(const SamplingParams& params);

  //按参数选择 greedy / temperature / top-k
  int Sample(const Tensor& logits, const SamplingParams& params);

  static int SampleGreedy(const Tensor& logits);
  int SampleTemperature(const Tensor& logits, float temperature);
  int SampleTopK(const Tensor& logits, float temperature, int top_k);

 private:
  std::mt19937 rng_;
};

//兼容旧调用：直接 ArgMax
int SampleGreedy(const Tensor& logits);

}

#endif
