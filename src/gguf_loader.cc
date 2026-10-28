//GGUF映射加载：metadata填ModelConfig，张量名映射后读进MiniLlamaModel
//线性层保留量化块，embedding 和归一化权重解码为 F32。
#include "mini_llama/gguf_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {
namespace {

//量化块大小，和ggml一致
constexpr int kQuantBlock = 32;

//Q8_0一块：fp16 scale + 32个int8
struct BlockQ80 {
  uint16_t d;
  int8_t qs[32];
};
static_assert(sizeof(BlockQ80) == 34, "Q8_0 block size");

//Q4_0一块：fp16 scale + 16字节装32个4bit
struct BlockQ40 {
  uint16_t d;
  uint8_t qs[16];
};
static_assert(sizeof(BlockQ40) == 18, "Q4_0 block size");

//Q4_1一块：fp16 scale + fp16 min + 16字节4bit
struct BlockQ41 {
  uint16_t d;
  uint16_t m;
  uint8_t qs[16];
};
static_assert(sizeof(BlockQ41) == 20, "Q4_1 block size");

//加载失败时返回空模型
MiniLlamaModel LoadFailure(const std::string& message) {
  MiniLlamaModel model;
  model.loaded = false;
  model.load_error = message;
  std::cerr << "Failed to load GGUF model: " << message << std::endl;
  return model;
}

  //IEEE754 half -> float，量化块里的scale要用这个函数
float Fp16ToFloat(uint16_t value) {
  uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      //次正规数：挪到正规再补指数差
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      uint32_t exp32 = exponent + (127 - 15);
      bits = sign | (exp32 << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    uint32_t exp32 = exponent + (127 - 15);
    bits = sign | (exp32 << 23) | (mantissa << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
  size_t total = 1;
  for (int dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error(std::string(caller) + ": invalid shape dim");
    }
    if (total > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      throw std::runtime_error(std::string(caller) + ": numel overflow");
    }
    total *= static_cast<size_t>(dim);
  }
  return total;
}

//llama.cpp风格名字转成我们内部名字；认不出返回空串
std::string MapTensorName(const std::string& gguf_name) {
  //词嵌入
  if (gguf_name == "token_embd.weight") {
    return "token_embedding";
  }
  //输出层norm
  if (gguf_name == "output_norm.weight") {
    return "final_norm";
  }
  //lm_head；没有时后面会和embedding绑权
  if (gguf_name == "output.weight") {
    return "lm_head";
  }

  //层内：blk.{L}.{op}.weight
  if (gguf_name.rfind("blk.", 0) != 0) {
    return "";
  }
  size_t dot1 = gguf_name.find('.', 4);
  if (dot1 == std::string::npos) {
    return "";
  }
  std::string layer_str = gguf_name.substr(4, dot1 - 4);
  size_t dot2 = gguf_name.find('.', dot1 + 1);
  if (dot2 == std::string::npos) {
    return "";
  }
  std::string op = gguf_name.substr(dot1 + 1, dot2 - dot1 - 1);
  std::string suffix = gguf_name.substr(dot2 + 1);
  if (suffix == "bias") {
    if (op == "attn_q") return "layers." + layer_str + ".bq";
    if (op == "attn_k") return "layers." + layer_str + ".bk";
    if (op == "attn_v") return "layers." + layer_str + ".bv";
    throw std::runtime_error("unsupported bias tensor: " + gguf_name);
  }
  if (suffix != "weight") {
    return "";
  }

  std::string mapped_op;
  if (op == "attn_norm") {
    mapped_op = "attention_norm";
  } else if (op == "attn_q") {
    mapped_op = "wq";
  } else if (op == "attn_k") {
    mapped_op = "wk";
  } else if (op == "attn_v") {
    mapped_op = "wv";
  } else if (op == "attn_output") {
    mapped_op = "wo";
  } else if (op == "ffn_norm") {
    mapped_op = "ffn_norm";
  } else if (op == "ffn_gate") {
    mapped_op = "w_gate";
  } else if (op == "ffn_up") {
    mapped_op = "w_up";
  } else if (op == "ffn_down") {
    mapped_op = "w_down";
  } else {
    return "";
  }
  return "layers." + layer_str + "." + mapped_op;
}

//GGUF二维权重常写成[ne0=in, ne1=out]，内存已是out行×in列
//我们的Tensor要[out, in]，所以两维时对调
std::vector<int> MapShape(const std::vector<uint64_t>& gguf_shape) {
  for (uint64_t dim : gguf_shape) {
    if (dim == 0 || dim > static_cast<uint64_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("GGUF tensor dimension out of range");
  }
  std::vector<int> shape;
  shape.reserve(gguf_shape.size());
  if (gguf_shape.size() == 2) {
    shape.push_back(static_cast<int>(gguf_shape[1]));
    shape.push_back(static_cast<int>(gguf_shape[0]));
    return shape;
  }
  for (uint64_t dim : gguf_shape) {
    shape.push_back(static_cast<int>(dim));
  }
  return shape;
}

//按行反量化Q8_0：x = d * q
Tensor DequantizeQ80(const std::vector<uint8_t>& raw,
                     const std::vector<int>& shape) {
  size_t total = CheckedNumel(shape, "DequantizeQ80");
  int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
  int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
  int row_blocks = (row_size + kQuantBlock - 1) / kQuantBlock;
  size_t expected_bytes =
      static_cast<size_t>(n_rows) * static_cast<size_t>(row_blocks) *
      sizeof(BlockQ80);
  if (raw.size() != expected_bytes) {
    throw std::runtime_error("Q8_0 size mismatch");
  }

  Tensor dst(shape, 0.0f);
  size_t block_idx = 0;
  for (int row = 0; row < n_rows; ++row) {
    int row_offset = row * row_size;
    for (int rb = 0; rb < row_blocks; ++rb) {
      int base = row_offset + rb * kQuantBlock;
      int k_end = std::min(base + kQuantBlock, row_offset + row_size);
      BlockQ80 block;
      std::memcpy(&block, raw.data() + block_idx * sizeof(BlockQ80),
                  sizeof(BlockQ80));
      ++block_idx;
      float d = Fp16ToFloat(block.d);
      for (int k = base; k < k_end; ++k) {
        dst.data[static_cast<size_t>(k)] =
            d * static_cast<float>(block.qs[k - base]);
      }
    }
  }
  return dst;
}

//Q4_0：半高半低各16个nibble，值域偏置-8
Tensor DequantizeQ40(const std::vector<uint8_t>& raw,
                     const std::vector<int>& shape) {
  size_t total = CheckedNumel(shape, "DequantizeQ40");
  int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
  int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
  int row_blocks = (row_size + kQuantBlock - 1) / kQuantBlock;
  size_t expected_bytes =
      static_cast<size_t>(n_rows) * static_cast<size_t>(row_blocks) *
      sizeof(BlockQ40);
  if (raw.size() != expected_bytes) {
    throw std::runtime_error("Q4_0 size mismatch");
  }

  Tensor dst(shape, 0.0f);
  size_t block_idx = 0;
  for (int row = 0; row < n_rows; ++row) {
    int row_offset = row * row_size;
    for (int rb = 0; rb < row_blocks; ++rb) {
      int base = row_offset + rb * kQuantBlock;
      int k_end = std::min(base + kQuantBlock, row_offset + row_size);
      BlockQ40 block;
      std::memcpy(&block, raw.data() + block_idx * sizeof(BlockQ40),
                  sizeof(BlockQ40));
      ++block_idx;
      float d = Fp16ToFloat(block.d);
      for (int j = 0; j < kQuantBlock / 2; ++j) {
        int idx0 = base + j;
        int idx1 = base + j + kQuantBlock / 2;
        int q0 = static_cast<int>(block.qs[j] & 0x0F) - 8;
        int q1 = static_cast<int>(block.qs[j] >> 4) - 8;
        if (idx0 < k_end) {
          dst.data[static_cast<size_t>(idx0)] = d * static_cast<float>(q0);
        }
        if (idx1 < k_end) {
          dst.data[static_cast<size_t>(idx1)] = d * static_cast<float>(q1);
        }
      }
    }
  }
  return dst;
}

//Q4_1：x = d * q + m，q在[0,15]
Tensor DequantizeQ41(const std::vector<uint8_t>& raw,
                     const std::vector<int>& shape) {
  size_t total = CheckedNumel(shape, "DequantizeQ41");
  int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
  int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
  int row_blocks = (row_size + kQuantBlock - 1) / kQuantBlock;
  size_t expected_bytes =
      static_cast<size_t>(n_rows) * static_cast<size_t>(row_blocks) *
      sizeof(BlockQ41);
  if (raw.size() != expected_bytes) {
    throw std::runtime_error("Q4_1 size mismatch");
  }

  Tensor dst(shape, 0.0f);
  size_t block_idx = 0;
  for (int row = 0; row < n_rows; ++row) {
    int row_offset = row * row_size;
    for (int rb = 0; rb < row_blocks; ++rb) {
      int base = row_offset + rb * kQuantBlock;
      int k_end = std::min(base + kQuantBlock, row_offset + row_size);
      BlockQ41 block;
      std::memcpy(&block, raw.data() + block_idx * sizeof(BlockQ41),
                  sizeof(BlockQ41));
      ++block_idx;
      float d = Fp16ToFloat(block.d);
      float m = Fp16ToFloat(block.m);
      for (int j = 0; j < kQuantBlock / 2; ++j) {
        int idx0 = base + j;
        int idx1 = base + j + kQuantBlock / 2;
        int q0 = static_cast<int>(block.qs[j] & 0x0F);
        int q1 = static_cast<int>(block.qs[j] >> 4);
        if (idx0 < k_end) {
          dst.data[static_cast<size_t>(idx0)] = d * static_cast<float>(q0) + m;
        }
        if (idx1 < k_end) {
          dst.data[static_cast<size_t>(idx1)] = d * static_cast<float>(q1) + m;
        }
      }
    }
  }
  return dst;
}

//F16整张量 -> F32
Tensor DequantizeF16(const std::vector<uint8_t>& raw,
                     const std::vector<int>& shape) {
  size_t total = CheckedNumel(shape, "DequantizeF16");
  if (raw.size() != total * sizeof(uint16_t)) {
    throw std::runtime_error("F16 size mismatch");
  }
  Tensor dst(shape, 0.0f);
  for (size_t i = 0; i < total; ++i) {
    uint16_t bits = 0;
    std::memcpy(&bits, raw.data() + i * sizeof(uint16_t), sizeof(bits));
    dst.data[i] = Fp16ToFloat(bits);
  }
  return dst;
}

//按GGML类型把原始字节变成F32 Tensor
Tensor DecodeTensor(const std::vector<uint8_t>& raw,
                    const std::vector<int>& shape, uint32_t ggml_type) {
  switch (ggml_type) {
    case kGgmlTypeF32: {
      size_t total = CheckedNumel(shape, "DecodeF32");
      size_t expected = total * sizeof(float);
      if (raw.size() != expected) {
        throw std::runtime_error("F32 size mismatch");
      }
      Tensor t(shape, 0.0f);
      std::memcpy(t.data.data(), raw.data(), expected);
      return t;
    }
    case kGgmlTypeF16:
      return DequantizeF16(raw, shape);
    case kGgmlTypeQ80:
      return DequantizeQ80(raw, shape);
    case kGgmlTypeQ40:
      return DequantizeQ40(raw, shape);
    case kGgmlTypeQ41:
      return DequantizeQ41(raw, shape);
    default:
      throw std::runtime_error("unsupported GGUF tensor type: " +
                               std::to_string(ggml_type));
  }
}

void RequireShape(const Tensor& tensor, const std::vector<int>& expected,
                  const std::string& name) {
  tensor.AssertShape(expected, name.c_str());
}

//装完后对一遍期望shape，防止映射/对调维搞反
void ValidateLoadedShapes(const MiniLlamaModel& model) {
  const ModelConfig& c = model.config;
  const int q_dim = c.n_heads * c.head_dim;
  const int kv_dim = c.n_kv_heads * c.head_dim;

  RequireShape(model.token_embedding, {c.vocab_size, c.dim},
               "GGUF token_embedding");
  RequireShape(model.final_norm, {c.dim}, "GGUF final_norm");
  RequireShape(model.lm_head, {c.vocab_size, c.dim}, "GGUF lm_head");

  if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
    throw std::runtime_error("GGUF layer count mismatch");
  }

  for (int layer = 0; layer < c.n_layers; ++layer) {
    const LayerWeights& lw = model.layers[static_cast<size_t>(layer)];
    const std::string prefix = "GGUF layers." + std::to_string(layer) + ".";
    RequireShape(lw.attention_norm, {c.dim}, prefix + "attention_norm");
    RequireShape(lw.wq, {q_dim, c.dim}, prefix + "wq");
    RequireShape(lw.wk, {kv_dim, c.dim}, prefix + "wk");
    RequireShape(lw.wv, {kv_dim, c.dim}, prefix + "wv");
    if (!lw.bq.data.empty()) RequireShape(lw.bq, {q_dim}, prefix + "bq");
    if (!lw.bk.data.empty()) RequireShape(lw.bk, {kv_dim}, prefix + "bk");
    if (!lw.bv.data.empty()) RequireShape(lw.bv, {kv_dim}, prefix + "bv");
    RequireShape(lw.wo, {c.dim, q_dim}, prefix + "wo");
    RequireShape(lw.ffn_norm, {c.dim}, prefix + "ffn_norm");
    RequireShape(lw.w_gate, {c.hidden_dim, c.dim}, prefix + "w_gate");
    RequireShape(lw.w_up, {c.hidden_dim, c.dim}, prefix + "w_up");
    RequireShape(lw.w_down, {c.dim, c.hidden_dim}, prefix + "w_down");
  }
}

//从metadata拼ModelConfig；优先qwen2.*，没有再试llama.*
ModelConfig BuildConfig(const std::string& path) {
  ModelConfig config;
  std::string architecture;
  if (!GgufGetMetadataString(path, "general.architecture", architecture) ||
      (architecture != "llama" && architecture != "qwen2")) {
    throw std::runtime_error("unsupported GGUF architecture: " + architecture);
  }
  config.rope_type = architecture == "qwen2" ? RopeType::kNeoX : RopeType::kNormal;

  auto get_u32 = [&](const std::string& key, int fallback) -> int {
    int64_t v = 0;
    if (GgufGetMetadataInt(path, key, v)) {
      if (v < 0 || v > std::numeric_limits<int>::max()) {
        throw std::runtime_error("GGUF integer out of range: " + key);
      }
      return static_cast<int>(v);
    }
    return fallback;
  };

  auto get_f32 = [&](const std::string& key, float fallback) -> float {
    double v = 0.0;
    if (GgufGetMetadataFloat(path, key, v)) {
      return static_cast<float>(v);
    }
    return fallback;
  };

  config.n_layers =
      get_u32("qwen2.block_count", get_u32("llama.block_count", 0));
  config.dim =
      get_u32("qwen2.embedding_length", get_u32("llama.embedding_length", 0));
  config.hidden_dim = get_u32("qwen2.feed_forward_length",
                              get_u32("llama.feed_forward_length", 0));
  config.n_heads = get_u32("qwen2.attention.head_count",
                           get_u32("llama.attention.head_count", 0));
  config.n_kv_heads = get_u32("qwen2.attention.head_count_kv",
                              get_u32("llama.attention.head_count_kv", 0));
  config.max_seq_len =
      get_u32("qwen2.context_length", get_u32("llama.context_length", 0));
  config.rope_theta = get_f32("qwen2.rope.freq_base",
                              get_f32("llama.rope.freq_base", 10000.0f));
  config.rms_norm_eps =
      get_f32("qwen2.attention.layer_norm_rms_epsilon",
              get_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f));

  //词表大小优先用tokenizer.ggml.tokens数组长度
  std::vector<std::string> tokens;
  if (GgufGetMetadataStringArray(path, "tokenizer.ggml.tokens", tokens)) {
    config.vocab_size = static_cast<int>(tokens.size());
  }

  if (config.n_layers <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
      config.n_heads <= 0 || config.n_kv_heads <= 0 ||
      config.max_seq_len <= 0 || config.vocab_size <= 0) {
    throw std::runtime_error(
        "invalid model config from GGUF: dim=" + std::to_string(config.dim) +
        " layers=" + std::to_string(config.n_layers) +
        " heads=" + std::to_string(config.n_heads) +
        " vocab=" + std::to_string(config.vocab_size));
  }
  if (config.dim % config.n_heads != 0) {
    throw std::runtime_error("dim not divisible by n_heads");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::runtime_error("n_heads not divisible by n_kv_heads");
  }

  config.head_dim = config.dim / config.n_heads;
  if (config.head_dim % 2 != 0 || !std::isfinite(config.rope_theta) ||
      config.rope_theta <= 0 || !std::isfinite(config.rms_norm_eps) ||
      config.rms_norm_eps <= 0) {
    throw std::runtime_error("invalid RoPE dimension, frequency base or norm epsilon");
  }
  // 这两种结构只支持完整 head 的普通 RoPE，不默默忽略缩放配置。
  int rotary_dim = get_u32(architecture + ".rope.dimension_count", config.head_dim);
  std::string scaling;
  if (rotary_dim != config.head_dim ||
      (GgufGetMetadataString(path, architecture + ".rope.scaling.type", scaling) &&
       scaling != "none")) {
    throw std::runtime_error("unsupported partial or scaled RoPE");
  }
  if (get_u32(architecture + ".attention.sliding_window", 0) > 0) {
    throw std::runtime_error("sliding-window attention is not supported");
  }

  std::cout << "GGUF config: vocab=" << config.vocab_size
            << " dim=" << config.dim << " layers=" << config.n_layers
            << " heads=" << config.n_heads << " kv_heads=" << config.n_kv_heads
            << " head_dim=" << config.head_dim
            << " hidden_dim=" << config.hidden_dim
            << " max_seq_len=" << config.max_seq_len
            << " rope_theta=" << config.rope_theta
            << " rms_norm_eps=" << config.rms_norm_eps << std::endl;

  return config;
}

}

//对外入口：解析GGUF -> 填config -> 按映射读权重
MiniLlamaModel LoadGgufModel(const std::string& gguf_path) {
  GgufReader reader;
  if (!reader.Load(gguf_path)) {
    return LoadFailure("GGUF parse error: " + reader.load_error);
  }

  MiniLlamaModel model;
  try {
    model.config = BuildConfig(gguf_path);
  } catch (const std::exception& e) {
    return LoadFailure(std::string("Failed to build config: ") + e.what());
  }

  const ModelConfig& c = model.config;

  //先建「内部名 -> tensor info」索引，跳过不认识的张量
  std::map<std::string, const GgufTensorInfo*> tensor_map;
  try {
   for (const auto& t : reader.tensors) {
    std::string mapped = MapTensorName(t.name);
    if (mapped.empty()) {
      continue;
    }
    if (!tensor_map.emplace(mapped, &t).second) {
      throw std::runtime_error("duplicate tensor mapping: " + mapped);
    }
   }
  } catch (const std::exception& e) {
    return LoadFailure(e.what());
  }

  //按映射名读一块，并解码成F32
  auto load_tensor = [&](const std::string& mapped_name) -> Tensor {
    auto it = tensor_map.find(mapped_name);
    if (it == tensor_map.end()) {
      throw std::runtime_error("missing tensor: " + mapped_name);
    }
    const GgufTensorInfo* info = it->second;
    std::vector<int> shape = MapShape(info->shape);
    std::vector<uint8_t> raw =
        ReadGgufTensorRaw(gguf_path, *info, reader.data_offset);
    if (raw.empty()) {
      throw std::runtime_error("failed to read tensor: " + mapped_name);
    }
    const bool linear = mapped_name == "lm_head" ||
        (mapped_name.rfind("layers.", 0) == 0 && mapped_name.find(".w") != std::string::npos);
    if (linear && shape.size() == 2 &&
        (info->type == kGgmlTypeQ80 || info->type == kGgmlTypeQ40 || info->type == kGgmlTypeQ41)) {
      if (shape[1] % 32 != 0) throw std::runtime_error("GGUF quantized row is not block aligned");
      return TensorFromQuantized(std::move(raw), shape[0], shape[1], static_cast<QuantType>(info->type));
    }
    return DecodeTensor(raw, shape, info->type);
  };

  try {
    model.token_embedding = load_tensor("token_embedding");

    model.layers.resize(static_cast<size_t>(c.n_layers));
    for (int layer = 0; layer < c.n_layers; ++layer) {
      LayerWeights& lw = model.layers[static_cast<size_t>(layer)];
      const std::string prefix = "layers." + std::to_string(layer) + ".";
      lw.attention_norm = load_tensor(prefix + "attention_norm");
      lw.wq = load_tensor(prefix + "wq");
      lw.wk = load_tensor(prefix + "wk");
      lw.wv = load_tensor(prefix + "wv");
      const bool needs_bias = c.rope_type == RopeType::kNeoX;
      if (needs_bias || tensor_map.count(prefix + "bq")) lw.bq = load_tensor(prefix + "bq");
      if (needs_bias || tensor_map.count(prefix + "bk")) lw.bk = load_tensor(prefix + "bk");
      if (needs_bias || tensor_map.count(prefix + "bv")) lw.bv = load_tensor(prefix + "bv");
      lw.wo = load_tensor(prefix + "wo");
      lw.ffn_norm = load_tensor(prefix + "ffn_norm");
      lw.w_gate = load_tensor(prefix + "w_gate");
      lw.w_up = load_tensor(prefix + "w_up");
      lw.w_down = load_tensor(prefix + "w_down");
    }

    model.final_norm = load_tensor("final_norm");

    //没有独立lm_head就绑权：直接拷贝embedding
    if (tensor_map.find("lm_head") != tensor_map.end()) {
      model.lm_head = load_tensor("lm_head");
    } else {
      model.lm_head = model.token_embedding;
    }

    ValidateLoadedShapes(model);
  } catch (const std::exception& e) {
    return LoadFailure(std::string("Failed to load GGUF weights: ") + e.what());
  }

  model.loaded = true;
  std::cout << "GGUF model loaded successfully from " << gguf_path << std::endl;
  return model;
}

}
