//把tiny的JSON+BIN模型临时写成一份完整GGUF，给gguf_loader单测用
#ifndef TESTS_GGUF_LOADER_FIXTURE_H_
#define TESTS_GGUF_LOADER_FIXTURE_H_

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"

namespace mini_llama {
namespace test {

//一条待写入的tensor：shape仍是内部的[out, in]，写文件时再对调
struct GgufTensorFixture {
  std::string gguf_name;
  std::vector<int> shape;
  const float* data = nullptr;
  size_t data_size = 0;
  uint64_t offset = 0;//相对data区的起点，后面按顺序排
};

//临时文件放系统temp，测完由调用方删掉
inline std::filesystem::path GgufLoaderTempPath(const std::string& name) {
  return std::filesystem::temp_directory_path() / ("mini_ai_" + name);
}

//小端uint32，和C++ reader按同样字节序读
inline void AppendU32Le(std::vector<char>& data, uint32_t value) {
  data.push_back(static_cast<char>(value & 0xff));
  data.push_back(static_cast<char>((value >> 8) & 0xff));
  data.push_back(static_cast<char>((value >> 16) & 0xff));
  data.push_back(static_cast<char>((value >> 24) & 0xff));
}

//小端uint64
inline void AppendU64Le(std::vector<char>& data, uint64_t value) {
  for (size_t i = 0; i < 8; ++i) {
    data.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
  }
}

//float按位拷成小端uint32再写，避免转成文本
inline void AppendF32Le(std::vector<char>& data, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  AppendU32Le(data, bits);
}

//GGUF字符串：先写长度，再写字节，不是\0结尾
inline void AppendString(std::vector<char>& data, const std::string& value) {
  AppendU64Le(data, value.size());
  data.insert(data.end(), value.begin(), value.end());
}

//一条uint32 metadata：key + type + value
inline void AppendMetadataU32(std::vector<char>& data, const std::string& key,
                              uint32_t value) {
  AppendString(data, key);
  AppendU32Le(data, kGgufTypeUint32);
  AppendU32Le(data, value);
}

//一条float32 metadata，给rope_theta和rms_norm_eps用
inline void AppendMetadataF32(std::vector<char>& data, const std::string& key,
                              float value) {
  AppendString(data, key);
  AppendU32Le(data, kGgufTypeFloat32);
  AppendF32Le(data, value);
}

//一条字符串metadata，比如general.architecture
inline void AppendMetadataString(std::vector<char>& data,
                                 const std::string& key,
                                 const std::string& value) {
  AppendString(data, key);
  AppendU32Le(data, kGgufTypeString);
  AppendString(data, value);
}

//写假词表，只为了让vocab_size等于数组长度
inline void AppendMetadataTokens(std::vector<char>& data, int vocab_size) {
  AppendString(data, "tokenizer.ggml.tokens");
  AppendU32Le(data, kGgufTypeArray);
  AppendU32Le(data, kGgufTypeString);
  AppendU64Le(data, static_cast<uint64_t>(vocab_size));
  for (int i = 0; i < vocab_size; ++i) {
    AppendString(data, "tok_" + std::to_string(i));
  }
}

//二维权重写入GGUF时用[ne0=in, ne1=out]，和MapShape对调互逆
inline std::vector<uint64_t> GgufShapeForFixture(
    const GgufTensorFixture& fixture) {
  if (fixture.shape.size() == 2) {
    return {
        static_cast<uint64_t>(fixture.shape[1]),
        static_cast<uint64_t>(fixture.shape[0]),
    };
  }
  std::vector<uint64_t> shape;
  shape.reserve(fixture.shape.size());
  for (int dim : fixture.shape) {
    shape.push_back(static_cast<uint64_t>(dim));
  }
  return shape;
}

//tensor目录：名字 + 维数 + 各维 + F32 + 相对offset，还不是权重本身
inline void AppendTensorInfo(std::vector<char>& data,
                             const GgufTensorFixture& fixture) {
  AppendString(data, fixture.gguf_name);
  std::vector<uint64_t> shape = GgufShapeForFixture(fixture);
  AppendU32Le(data, static_cast<uint32_t>(shape.size()));
  for (uint64_t dim : shape) {
    AppendU64Le(data, dim);
  }
  AppendU32Le(data, kGgmlTypeF32);
  AppendU64Le(data, fixture.offset);
}

//把float权重原样追加到data区
inline void AppendTensorBytes(std::vector<char>& data,
                              const GgufTensorFixture& fixture) {
  const char* bytes = reinterpret_cast<const char*>(fixture.data);
  size_t n_bytes = fixture.data_size * sizeof(float);
  data.insert(data.end(), bytes, bytes + n_bytes);
}

//一层权重量成GGUF名字：wq写成blk.L.attn_q.weight
inline void AddLayerTensors(std::vector<GgufTensorFixture>& tensors,
                            const MiniLlamaModel& model, int layer) {
  const LayerWeights& lw = model.layers[static_cast<size_t>(layer)];
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  tensors.push_back({prefix + "attn_norm.weight", lw.attention_norm.shape,
                     lw.attention_norm.data.data(),
                     lw.attention_norm.data.size(), 0});
  tensors.push_back({prefix + "attn_q.weight", lw.wq.shape, lw.wq.data.data(),
                     lw.wq.data.size(), 0});
  tensors.push_back({prefix + "attn_k.weight", lw.wk.shape, lw.wk.data.data(),
                     lw.wk.data.size(), 0});
  tensors.push_back({prefix + "attn_v.weight", lw.wv.shape, lw.wv.data.data(),
                     lw.wv.data.size(), 0});
  tensors.push_back({prefix + "attn_output.weight", lw.wo.shape,
                     lw.wo.data.data(), lw.wo.data.size(), 0});
  tensors.push_back({prefix + "ffn_norm.weight", lw.ffn_norm.shape,
                     lw.ffn_norm.data.data(), lw.ffn_norm.data.size(), 0});
  tensors.push_back({prefix + "ffn_gate.weight", lw.w_gate.shape,
                     lw.w_gate.data.data(), lw.w_gate.data.size(), 0});
  tensors.push_back({prefix + "ffn_up.weight", lw.w_up.shape,
                     lw.w_up.data.data(), lw.w_up.data.size(), 0});
  tensors.push_back({prefix + "ffn_down.weight", lw.w_down.shape,
                     lw.w_down.data.data(), lw.w_down.data.size(), 0});
}

//从models/tiny生成一份可被LoadGgufModel吃下的临时GGUF
inline std::filesystem::path WriteTinyLlamaGgufFixture(
    const std::string& name) {
  MiniLlamaModel source =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!source.loaded) {
    throw std::runtime_error("failed to load tiny source model: " +
                             source.load_error);
  }

  //先列目录：embedding、每层attn/ffn、output_norm、lm_head
  std::vector<GgufTensorFixture> tensors;
  tensors.push_back({"token_embd.weight", source.token_embedding.shape,
                     source.token_embedding.data.data(),
                     source.token_embedding.data.size(), 0});
  for (int layer = 0; layer < source.config.n_layers; ++layer) {
    AddLayerTensors(tensors, source, layer);
  }
  tensors.push_back({"output_norm.weight", source.final_norm.shape,
                     source.final_norm.data.data(),
                     source.final_norm.data.size(), 0});
  tensors.push_back({"output.weight", source.lm_head.shape,
                     source.lm_head.data.data(), source.lm_head.data.size(),
                     0});

  //相对offset按float字节紧挨着排，中间不留洞
  uint64_t offset = 0;
  for (GgufTensorFixture& tensor : tensors) {
    tensor.offset = offset;
    offset += static_cast<uint64_t>(tensor.data_size * sizeof(float));
  }

  constexpr uint64_t kAlignment = 32;
  constexpr uint64_t kMetadataCount = 12;

  //Header：magic + v3 + tensor数 + 下面12条metadata
  std::vector<char> data;
  data.insert(data.end(), {'G', 'G', 'U', 'F'});
  AppendU32Le(data, 3);
  AppendU64Le(data, static_cast<uint64_t>(tensors.size()));
  AppendU64Le(data, kMetadataCount);

  AppendMetadataString(data, "general.architecture", "llama");
  AppendMetadataString(data, "general.name", "mini-ai-tiny-loader-test");
  AppendMetadataU32(data, "general.alignment",
                    static_cast<uint32_t>(kAlignment));
  AppendMetadataU32(data, "llama.block_count",
                    static_cast<uint32_t>(source.config.n_layers));
  AppendMetadataU32(data, "llama.context_length",
                    static_cast<uint32_t>(source.config.max_seq_len));
  AppendMetadataU32(data, "llama.embedding_length",
                    static_cast<uint32_t>(source.config.dim));
  AppendMetadataU32(data, "llama.feed_forward_length",
                    static_cast<uint32_t>(source.config.hidden_dim));
  AppendMetadataU32(data, "llama.attention.head_count",
                    static_cast<uint32_t>(source.config.n_heads));
  AppendMetadataU32(data, "llama.attention.head_count_kv",
                    static_cast<uint32_t>(source.config.n_kv_heads));
  AppendMetadataF32(data, "llama.rope.freq_base", source.config.rope_theta);
  AppendMetadataF32(data, "llama.attention.layer_norm_rms_epsilon",
                    source.config.rms_norm_eps);
  AppendMetadataTokens(data, source.config.vocab_size);

  for (const GgufTensorFixture& tensor : tensors) {
    AppendTensorInfo(data, tensor);
  }

  //info长度不是32的倍数时补0，再写真正的权重
  size_t padding = static_cast<size_t>(
      (kAlignment - (data.size() % kAlignment)) % kAlignment);
  data.insert(data.end(), padding, '\0');
  for (const GgufTensorFixture& tensor : tensors) {
    AppendTensorBytes(data, tensor);
  }

  std::filesystem::path path = GgufLoaderTempPath(name);
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to create GGUF fixture: " + path.string());
  }
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  return path;
}

}
}

#endif
