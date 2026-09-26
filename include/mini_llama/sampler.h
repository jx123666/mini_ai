#ifndef INCLUDE_MINI_LLAMA_SAMPLER_H_
#define INCLUDE_MINI_LLAMA_SAMPLER_H_

namespace mini_llama {

struct SamplingParams {
  float temperature = 1.0f;
  int top_k = 0;
  int seed = 0;
};

}

#endif
