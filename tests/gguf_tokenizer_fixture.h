//写一份只含tokenizer metadata的小GGUF，给接入测试用
#ifndef TESTS_GGUF_TOKENIZER_FIXTURE_H_
#define TESTS_GGUF_TOKENIZER_FIXTURE_H_

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"

namespace mini_llama {
namespace test {

//临时文件放系统temp，测完由调用方删掉
inline std::filesystem::path GgufTokenizerTempPath(const std::string& name) {
  return std::filesystem::temp_directory_path() / ("mini_ai_" + name);
}

//小端uint32，和C++ reader按同样字节序读
inline void TokAppendU32Le(std::vector<char>& data, uint32_t value) {
  data.push_back(static_cast<char>(value & 0xff));
  data.push_back(static_cast<char>((value >> 8) & 0xff));
  data.push_back(static_cast<char>((value >> 16) & 0xff));
  data.push_back(static_cast<char>((value >> 24) & 0xff));
}

//小端uint64
inline void TokAppendU64Le(std::vector<char>& data, uint64_t value) {
  for (size_t i = 0; i < 8; ++i) {
    data.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
  }
}

//GGUF字符串：先写长度，再写字节，不是\0结尾
inline void TokAppendString(std::vector<char>& data, const std::string& value) {
  TokAppendU64Le(data, value.size());
  data.insert(data.end(), value.begin(), value.end());
}

//一条字符串metadata，比如general.architecture
inline void TokAppendMetadataString(std::vector<char>& data,
                                    const std::string& key,
                                    const std::string& value) {
  TokAppendString(data, key);
  TokAppendU32Le(data, kGgufTypeString);
  TokAppendString(data, value);
}

//一条uint32 metadata，给alignment和bos/eos用
inline void TokAppendMetadataU32(std::vector<char>& data, const std::string& key,
                                 uint32_t value) {
  TokAppendString(data, key);
  TokAppendU32Le(data, kGgufTypeUint32);
  TokAppendU32Le(data, value);
}

//字符串数组：tokens和merges都是这种
inline void TokAppendStringArray(std::vector<char>& data, const std::string& key,
                                 const std::vector<std::string>& values) {
  TokAppendString(data, key);
  TokAppendU32Le(data, kGgufTypeArray);
  TokAppendU32Le(data, kGgufTypeString);
  TokAppendU64Le(data, static_cast<uint64_t>(values.size()));
  for (const std::string& value : values) {
    TokAppendString(data, value);
  }
}

//int32数组，token_type用它标哪些是control
inline void TokAppendI32Array(std::vector<char>& data, const std::string& key,
                              const std::vector<int32_t>& values) {
  TokAppendString(data, key);
  TokAppendU32Le(data, kGgufTypeArray);
  TokAppendU32Le(data, kGgufTypeInt32);
  TokAppendU64Le(data, static_cast<uint64_t>(values.size()));
  for (int32_t value : values) {
    TokAppendU32Le(data, static_cast<uint32_t>(value));
  }
}

//词表里 a+b 能合成 ab；<|im_start|> 标成control(type=3)
inline std::filesystem::path WriteTinyGgufTokenizerFixture(
    const std::string& name) {
  //0/1/2是普通片段，后面四个type=3是special，编码时整段吃掉
  const std::vector<std::string> tokens = {
      "a", "b", "ab", "<s>", "</s>", "<|im_start|>", "<|im_end|>",
  };
  const std::vector<int32_t> token_types = {1, 1, 1, 3, 3, 3, 3};
  const std::vector<std::string> merges = {"a b"};

  //Header：magic + v3 + 0个tensor + 8条metadata
  std::vector<char> data;
  data.insert(data.end(), {'G', 'G', 'U', 'F'});
  TokAppendU32Le(data, 3);
  TokAppendU64Le(data, 0);  //没有tensor
  TokAppendU64Le(data, 8);  //8条metadata

  TokAppendMetadataString(data, "general.architecture", "qwen2");
  TokAppendMetadataString(data, "general.name", "tiny-tokenizer-test");
  TokAppendMetadataU32(data, "general.alignment", 32);
  TokAppendStringArray(data, "tokenizer.ggml.tokens", tokens);
  TokAppendI32Array(data, "tokenizer.ggml.token_type", token_types);
  TokAppendStringArray(data, "tokenizer.ggml.merges", merges);
  TokAppendMetadataU32(data, "tokenizer.ggml.bos_token_id", 3);
  TokAppendMetadataU32(data, "tokenizer.ggml.eos_token_id", 4);

  //对齐到32，这份文件没有权重，padding后面就结束
  size_t padding = (32 - (data.size() % 32)) % 32;
  data.insert(data.end(), padding, '\0');

  std::filesystem::path path = GgufTokenizerTempPath(name);
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to create tokenizer fixture: " +
                             path.string());
  }
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  return path;
}

}
}

#endif
