#include "mini_llama/backend.h"
#include <stdexcept>

#ifndef MINI_AI_CUDA
namespace mini_llama {
void EnableCuda(MiniLlamaModel &, int) {
  throw std::runtime_error(
      "CUDA is unavailable in this build; configure with -DMINI_AI_CUDA=ON");
}
std::string CudaDeviceInfo() { return "CUDA is unavailable in this build"; }
std::string CudaStats(const MiniLlamaModel &, const MiniLlamaContext &) {
  return "";
}
Tensor ForwardCuda(MiniLlamaContext &, const MiniLlamaModel &, int) {
  throw std::runtime_error("CUDA is unavailable in this build");
}
} // namespace mini_llama
#endif
