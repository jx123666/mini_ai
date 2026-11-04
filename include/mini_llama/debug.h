#ifndef INCLUDE_MINI_LLAMA_DEBUG_H_
#define INCLUDE_MINI_LLAMA_DEBUG_H_

#include "mini_llama/tensor.h"
#include <string>

namespace mini_llama {
void DumpLogits(const std::string &directory, int step, const Tensor &logits);
std::string FormatTopLogits(const Tensor &logits, int count = 5);
double TokensPerSecond(int tokens, double milliseconds);
} // namespace mini_llama
#endif
