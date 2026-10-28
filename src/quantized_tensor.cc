#include "mini_llama/quantized_tensor.h"
#include "mini_llama/model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace mini_llama {
namespace {
size_t BlockBytes(QuantType type) {
  switch (type) {
  case QuantType::kQ80:
    return 34;
  case QuantType::kQ40:
    return 18;
  case QuantType::kQ41:
    return 20;
  }
  throw std::invalid_argument("unknown quantization type");
}
uint16_t ReadHalf(const uint8_t *p) {
  return uint16_t(p[0]) | uint16_t(p[1]) << 8;
}
void WriteHalf(uint8_t *p, float value) {
  const uint16_t bits = FloatToHalf(value);
  p[0] = bits & 255;
  p[1] = bits >> 8;
}
float Value(const uint8_t *block, QuantType type, int i) {
  const float scale = HalfToFloat(ReadHalf(block));
  if (type == QuantType::kQ80) {
    const int q = block[2 + i] < 128 ? block[2 + i] : int(block[2 + i]) - 256;
    return scale * q;
  }
  const int offset = type == QuantType::kQ41 ? 4 : 2;
  const int packed = block[offset + i % 16];
  const int q = i < 16 ? packed & 15 : packed >> 4;
  return type == QuantType::kQ41 ? scale * q + HalfToFloat(ReadHalf(block + 2))
                                 : scale * (q - 8);
}
} // namespace

float HalfToFloat(uint16_t bits) {
  const float sign = bits & 0x8000 ? -1.0f : 1.0f;
  const int exponent = (bits >> 10) & 31;
  const int fraction = bits & 1023;
  if (exponent == 31)
    return fraction ? std::numeric_limits<float>::quiet_NaN()
                    : sign * std::numeric_limits<float>::infinity();
  return sign * (exponent ? std::ldexp(1.0f + fraction / 1024.0f, exponent - 15)
                          : std::ldexp(static_cast<float>(fraction), -24));
}

uint16_t FloatToHalf(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = (bits >> 16) & 0x8000;
  int exponent = int((bits >> 23) & 255) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffff;
  if (((bits >> 23) & 255) == 255)
    return sign | (mantissa ? 0x7e00 : 0x7c00);
  if (exponent >= 31)
    return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10)
      return sign;
    mantissa |= 0x800000;
    const int shift = 14 - exponent;
    const uint32_t rounded =
        (mantissa + ((1u << (shift - 1)) - 1) + ((mantissa >> shift) & 1)) >>
        shift;
    return sign | static_cast<uint16_t>(rounded);
  }
  mantissa += 0xfff + ((mantissa >> 13) & 1);
  if (mantissa & 0x800000) {
    mantissa = 0;
    ++exponent;
  }
  if (exponent >= 31)
    return sign | 0x7c00;
  return sign | static_cast<uint16_t>(exponent << 10) |
         static_cast<uint16_t>(mantissa >> 13);
}

Tensor TensorFromQuantized(std::vector<uint8_t> bytes, int rows, int cols,
                           QuantType type) {
  if (rows <= 0 || cols <= 0)
    throw std::invalid_argument("invalid quantized shape");
  const size_t blocks = (static_cast<size_t>(cols) + 31) / 32;
  if (blocks > std::numeric_limits<size_t>::max() / BlockBytes(type) / rows ||
      bytes.size() != blocks * BlockBytes(type) * rows) {
    throw std::runtime_error("quantized storage size mismatch");
  }
  Tensor out;
  out.shape = {rows, cols};
  out.quantized = std::make_shared<QuantizedStorage>(
      QuantizedStorage{type, rows, cols, std::move(bytes)});
  return out;
}

Tensor QuantizeTensor(const Tensor &tensor, QuantType type) {
  if (tensor.num_dims() != 2)
    throw std::invalid_argument("quantization requires a matrix");
  const Tensor decoded = tensor.quantized ? DequantizeTensor(tensor) : Tensor{};
  const Tensor &source = tensor.quantized ? decoded : tensor;
  const int rows = source.shape[0], cols = source.shape[1];
  const size_t block_bytes = BlockBytes(type),
               blocks = (static_cast<size_t>(cols) + 31) / 32;
  std::vector<uint8_t> bytes(static_cast<size_t>(rows) * blocks * block_bytes);
  for (int row = 0; row < rows; ++row) {
    for (size_t b = 0; b < blocks; ++b) {
      float values[32] = {};
      float min = 0, max = 0, signed_max = 0;
      for (int i = 0; i < 32 && b * 32 + i < static_cast<size_t>(cols); ++i) {
        const float value = source[row * cols + b * 32 + i];
        if (!std::isfinite(value))
          throw std::runtime_error("non-finite quantization input");
        values[i] = value;
        min = std::min(min, value);
        max = std::max(max, value);
        if (std::fabs(value) > std::fabs(signed_max))
          signed_max = value;
      }
      uint8_t *block =
          bytes.data() + (static_cast<size_t>(row) * blocks + b) * block_bytes;
      float scale = type == QuantType::kQ80   ? std::fabs(signed_max) / 127.0f
                    : type == QuantType::kQ40 ? signed_max / -8.0f
                                              : (max - min) / 15.0f;
      WriteHalf(block, scale);
      scale = HalfToFloat(ReadHalf(block));
      if (!std::isfinite(scale))
        throw std::runtime_error("quantization scale exceeds F16 range");
      if (type == QuantType::kQ41)
        WriteHalf(block + 2, min);
      const float offset =
          type == QuantType::kQ41 ? HalfToFloat(ReadHalf(block + 2)) : 0;
      if (!std::isfinite(offset))
        throw std::runtime_error("quantization offset exceeds F16 range");
      for (int i = 0; i < 32; ++i) {
        int q =
            scale == 0
                ? 0
                : static_cast<int>(std::round((values[i] - offset) / scale));
        if (type == QuantType::kQ80)
          block[2 + i] = static_cast<uint8_t>(std::clamp(q, -127, 127));
        else {
          q = type == QuantType::kQ40 ? std::clamp(q + 8, 0, 15)
                                      : std::clamp(q, 0, 15);
          const int base = type == QuantType::kQ41 ? 4 : 2;
          block[base + i % 16] |= static_cast<uint8_t>(q << (i < 16 ? 0 : 4));
        }
      }
    }
  }
  return TensorFromQuantized(std::move(bytes), rows, cols, type);
}

float QuantizedDotRow(const QuantizedStorage &storage, int row,
                      const float *x) {
  const size_t block_bytes = BlockBytes(storage.type);
  const size_t blocks = (static_cast<size_t>(storage.cols) + 31) / 32;
  float sum = 0;
  // 逐块解码后立即累加，整个权重矩阵始终保留压缩格式。
  for (size_t b = 0; b < blocks; ++b) {
    const auto *block = storage.bytes.data() +
                        (static_cast<size_t>(row) * blocks + b) * block_bytes;
    const float scale = HalfToFloat(ReadHalf(block));
    const float offset =
        storage.type == QuantType::kQ41 ? HalfToFloat(ReadHalf(block + 2)) : 0;
    for (int i = 0; i < 32 && b * 32 + i < static_cast<size_t>(storage.cols);
         ++i) {
      float value;
      if (storage.type == QuantType::kQ80) {
        const int q =
            block[2 + i] < 128 ? block[2 + i] : int(block[2 + i]) - 256;
        value = scale * q;
      } else {
        const int packed =
            block[(storage.type == QuantType::kQ41 ? 4 : 2) + i % 16];
        const int q = i < 16 ? packed & 15 : packed >> 4;
        value = storage.type == QuantType::kQ41 ? scale * q + offset
                                                : scale * (q - 8);
      }
      sum += value * x[b * 32 + i];
    }
  }
  return sum;
}

Tensor DequantizeTensor(const Tensor &tensor) {
  if (!tensor.quantized)
    return tensor;
  const auto &s = *tensor.quantized;
  Tensor out(tensor.shape);
  const size_t block_bytes = BlockBytes(s.type),
               blocks = (static_cast<size_t>(s.cols) + 31) / 32;
  for (int row = 0; row < s.rows; ++row)
    for (int col = 0; col < s.cols; ++col)
      out[row * s.cols + col] =
          Value(s.bytes.data() + (row * blocks + col / 32) * block_bytes,
                s.type, col % 32);
  return out;
}

void QuantizeModel(MiniLlamaModel &model, QuantType type) {
  if (model.cuda)
    throw std::runtime_error(
        "quantize weights before uploading the model to CUDA");
  for (auto &layer : model.layers) {
    for (Tensor *weight : {&layer.wq, &layer.wk, &layer.wv, &layer.wo,
                           &layer.w_gate, &layer.w_up, &layer.w_down}) {
      *weight = QuantizeTensor(*weight, type);
    }
  }
  model.lm_head = QuantizeTensor(model.lm_head, type);
}
} // namespace mini_llama
