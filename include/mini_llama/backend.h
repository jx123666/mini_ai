#ifndef INCLUDE_MINI_LLAMA_BACKEND_H_
#define INCLUDE_MINI_LLAMA_BACKEND_H_

#include "mini_llama/model.h"
#include <string>

namespace mini_llama {
struct MiniLlamaContext;
void EnableCuda(MiniLlamaModel &model, int device = 0);
std::string CudaDeviceInfo();
std::string CudaStats(const MiniLlamaModel &model,
                      const MiniLlamaContext &context);
Tensor ForwardCuda(MiniLlamaContext &context, const MiniLlamaModel &model,
                   int token);
} // namespace mini_llama
#endif
