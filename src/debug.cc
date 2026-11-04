#include "mini_llama/debug.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace mini_llama {
void DumpLogits(const std::string &directory, int step, const Tensor &logits) {
  if (directory.empty())
    return;
  std::filesystem::create_directories(directory);
  std::ostringstream name;
  name << "logits_" << std::setfill('0') << std::setw(4) << step << ".bin";
  std::ofstream file(std::filesystem::path(directory) / name.str(),
                     std::ios::binary);
  file.write(reinterpret_cast<const char *>(logits.data.data()),
             static_cast<std::streamsize>(logits.data.size() * sizeof(float)));
  if (!file)
    throw std::runtime_error("failed to write logits: " + name.str());
}

std::string FormatTopLogits(const Tensor &logits, int count) {
  if (count < 0)
    throw std::invalid_argument("negative top-logit count");
  std::vector<size_t> indices(logits.size());
  std::iota(indices.begin(), indices.end(), 0);
  count = std::min(count, static_cast<int>(indices.size()));
  std::partial_sort(indices.begin(), indices.begin() + count, indices.end(),
                    [&](size_t a, size_t b) { return logits[a] > logits[b]; });
  std::ostringstream out;
  for (int i = 0; i < count; ++i)
    out << (i ? " " : "") << indices[i] << ':' << logits[indices[i]];
  return out.str();
}

double TokensPerSecond(int tokens, double milliseconds) {
  return milliseconds > 0 ? tokens * 1000.0 / milliseconds : 0;
}
} // namespace mini_llama
