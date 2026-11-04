#ifndef INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_
#define INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_

#include <cstdint>
#include <vector>

namespace mini_llama {
struct Tensor;
struct MiniLlamaModel;
enum class QuantType : uint32_t { kQ80 = 8, kQ40 = 2, kQ41 = 3 };

struct QuantizedStorage {
  QuantType type;
  int rows;
  int cols;
  std::vector<uint8_t> bytes;
};

float HalfToFloat(uint16_t bits);
uint16_t FloatToHalf(float value);
Tensor QuantizeTensor(const Tensor &tensor, QuantType type);
Tensor DequantizeTensor(const Tensor &tensor);
Tensor TensorFromQuantized(std::vector<uint8_t> bytes, int rows, int cols,
                           QuantType type);
float QuantizedDotRow(const QuantizedStorage &storage, int row, const float *x);
void QuantizeModel(MiniLlamaModel &model, QuantType type);
} // namespace mini_llama
#endif
