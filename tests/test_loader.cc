//测 loader：解析清单、校验 tensor、拒绝坏配置、bin 尾部多余字节
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "mini_llama/loader.h"
#include "tests/test_main.h"

using mini_llama::LoadModel;
using mini_llama::MiniLlamaModel;
using mini_llama::ModelManifest;
using mini_llama::ParseManifest;

namespace {

std::string ReadTextFile(const std::string& path) {
  std::ifstream in(path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::filesystem::path TestTempPath(const std::string& name) {
  return std::filesystem::temp_directory_path() / ("mini_ai_" + name);
}

void WriteTextFile(const std::filesystem::path& path,
                   const std::string& content) {
  std::ofstream out(path);
  out << content;
}

bool ReplaceOnce(std::string& text, const std::string& from,
                 const std::string& to) {
  size_t pos = text.find(from);
  if (pos == std::string::npos) {
    return false;
  }
  text.replace(pos, from.size(), to);
  return true;
}

bool ReplaceAfter(std::string& text, const std::string& marker,
                  const std::string& from, const std::string& to) {
  size_t marker_pos = text.find(marker);
  if (marker_pos == std::string::npos) {
    return false;
  }
  size_t pos = text.find(from, marker_pos);
  if (pos == std::string::npos) {
    return false;
  }
  text.replace(pos, from.size(), to);
  return true;
}

bool ParseManifestFails(const std::string& json, const std::string& filename) {
  std::filesystem::path path = TestTempPath(filename);
  WriteTextFile(path, json);
  try {
    ParseManifest(path.string());
    std::filesystem::remove(path);
    return false;
  } catch (const std::runtime_error&) {
    std::filesystem::remove(path);
    return true;
  }
}

std::string ReplaceTokenizerType(const std::string& json,
                                 const std::string& tokenizer_type) {
  std::string updated = json;
  if (!ReplaceOnce(updated, "\"type\": \"json_vocab\"",
                   "\"type\": \"" + tokenizer_type + "\"")) {
    throw std::runtime_error("failed to replace tokenizer type");
  }
  return updated;
}

std::string RemoveTokenizerBlock(const std::string& json) {
  const std::string tokenizer_block =
      "  \"tokenizer\": {\n"
      "    \"type\": \"json_vocab\",\n"
      "    \"path\": \"vocab.json\",\n"
      "    \"bos_id\": 1,\n"
      "    \"eos_id\": 2,\n"
      "    \"unk_id\": 0\n"
      "  },\n";
  std::string updated = json;
  if (!ReplaceOnce(updated, tokenizer_block, "")) {
    throw std::runtime_error("failed to remove tokenizer block");
  }
  return updated;
}

}

//正常清单能解析出 config、tokenizer、tensor 元数据
static bool TestLoaderManifestParsesMetadata() {
  ModelManifest manifest = ParseManifest("models/tiny/model.json");
  MINI_LLAMA_ASSERT_EQ(manifest.config.vocab_size, 128);
  MINI_LLAMA_ASSERT_EQ(manifest.config.n_layers, 2);
  MINI_LLAMA_ASSERT_EQ(manifest.tensors.size(), 21);
  MINI_LLAMA_ASSERT_TRUE(manifest.tensors[0].name == "token_embedding");
  MINI_LLAMA_ASSERT_TRUE(manifest.tensors[0].dtype == "float32");
  MINI_LLAMA_ASSERT_EQ(manifest.tensors[0].offset, 0);
  MINI_LLAMA_ASSERT_EQ(manifest.tensors[0].byte_size, 16384);
  MINI_LLAMA_ASSERT_TRUE(manifest.tokenizer.type == "json_vocab");
  MINI_LLAMA_ASSERT_TRUE(manifest.tokenizer.path == "vocab.json");
  MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.bos_id, 1);
  MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.eos_id, 2);
  MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.unk_id, 0);
  return true;
}

//没有 tokenizer 块时默认 ascii
static bool TestLoaderManifestDefaultsAsciiWithoutTokenizerBlock() {
  std::string json = ReadTextFile("models/tiny/model.json");
  std::string without_tokenizer = RemoveTokenizerBlock(json);
  std::filesystem::path path = TestTempPath("tokenizer_manifest.json");
  WriteTextFile(path, without_tokenizer);

  ModelManifest manifest = ParseManifest(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(manifest.tokenizer.type == "ascii");
  MINI_LLAMA_ASSERT_TRUE(manifest.tokenizer.path.empty());
  MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.bos_id, 1);
  MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.eos_id, 2);
  MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.unk_id, 0);
  return true;
}

// JSON manifest 目前只接受 ascii 和 json_vocab。
static bool TestLoaderRejectsInvalidTokenizerType() {
  std::string json = ReadTextFile("models/tiny/model.json");
  std::string with_tokenizer = ReplaceTokenizerType(json, "bpe");
  MINI_LLAMA_ASSERT_TRUE(
      ParseManifestFails(with_tokenizer, "invalid_tokenizer_type.json"));
  return true;
}

//重名 tensor 不行
static bool TestLoaderRejectsDuplicateTensorName() {
  std::string json = ReadTextFile("models/tiny/model.json");
  MINI_LLAMA_ASSERT_TRUE(ReplaceOnce(json, "\"name\": \"final_norm\"",
                                     "\"name\": \"token_embedding\""));
  MINI_LLAMA_ASSERT_TRUE(ParseManifestFails(json, "duplicate_tensor.json"));
  return true;
}

//必填权重形状必须和 config 对得上
static bool TestLoaderRejectsRequiredShapeMismatch() {
  std::string json = ReadTextFile("models/tiny/model.json");
  MINI_LLAMA_ASSERT_TRUE(
      ReplaceAfter(json, "\"name\": \"lm_head\"",
                   "\"shape\": [\n        128,\n        32\n      ]",
                   "\"shape\": [\n        64,\n        64\n      ]"));
  MINI_LLAMA_ASSERT_TRUE(ParseManifestFails(json, "shape_mismatch.json"));
  return true;
}

//offset 区间不能重叠
static bool TestLoaderRejectsOverlappingTensorRanges() {
  std::string json = ReadTextFile("models/tiny/model.json");
  MINI_LLAMA_ASSERT_TRUE(ReplaceAfter(json, "\"name\": \"lm_head\"",
                                      "\"offset\": 115840",
                                      "\"offset\": 115712"));
  MINI_LLAMA_ASSERT_TRUE(ParseManifestFails(json, "overlap.json"));
  return true;
}

//缺 dtype 字段要拒绝
static bool TestLoaderRejectsMissingDtype() {
  std::string json = ReadTextFile("models/tiny/model.json");
  MINI_LLAMA_ASSERT_TRUE(ReplaceAfter(json, "\"name\": \"lm_head\"",
                                      "      \"dtype\": \"float32\",\n", ""));
  MINI_LLAMA_ASSERT_TRUE(ParseManifestFails(json, "missing_dtype.json"));
  return true;
}

//整数配置不能写成 4.5
static bool TestLoaderRejectsFractionalIntegerConfig() {
  std::string json = ReadTextFile("models/tiny/model.json");
  MINI_LLAMA_ASSERT_TRUE(
      ReplaceOnce(json, "\"n_heads\": 4", "\"n_heads\": 4.5"));
  MINI_LLAMA_ASSERT_TRUE(
      ParseManifestFails(json, "fractional_integer_config.json"));
  return true;
}

//浮点配置不能是 NaN
static bool TestLoaderRejectsNonfiniteFloatConfig() {
  std::string json = ReadTextFile("models/tiny/model.json");
  MINI_LLAMA_ASSERT_TRUE(
      ReplaceOnce(json, "\"rope_theta\": 10000.0", "\"rope_theta\": NaN"));
  MINI_LLAMA_ASSERT_TRUE(
      ParseManifestFails(json, "nonfinite_float_config.json"));
  return true;
}

//bin 末尾多字节说明文件对不上清单
static bool TestLoaderRejectsTrailingWeightBytes() {
  std::filesystem::path weights_path = TestTempPath("trailing_model.bin");
  {
    std::ifstream in("models/tiny/model.bin", std::ios::binary);
    std::ofstream out(weights_path, std::ios::binary);
    out << in.rdbuf();
    char extra = 0;
    out.write(&extra, 1);
  }

  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", weights_path.string());
  std::filesystem::remove(weights_path);
  MINI_LLAMA_ASSERT_TRUE(!model.loaded);
  MINI_LLAMA_ASSERT_TRUE(model.load_error.find("trailing bytes") !=
                         std::string::npos);
  return true;
}

//正常 JSON+BIN 能加载成功
static bool TestLoaderLoadsTinyModel() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  MINI_LLAMA_ASSERT_TRUE(model.loaded);
  MINI_LLAMA_ASSERT_EQ(model.config.vocab_size, 128);
  MINI_LLAMA_ASSERT_EQ(model.layers.size(), 2);
  MINI_LLAMA_ASSERT_EQ(model.token_embedding.shape[0], 128);
  MINI_LLAMA_ASSERT_EQ(model.token_embedding.shape[1], 32);
  MINI_LLAMA_ASSERT_EQ(model.lm_head.shape[0], 128);
  return true;
}

static struct LoaderTestRegistrar {
  LoaderTestRegistrar() {
    RegisterTest("loader_manifest_parses_metadata",
                 TestLoaderManifestParsesMetadata);
    RegisterTest("loader_manifest_defaults_ascii_without_tokenizer_block",
                 TestLoaderManifestDefaultsAsciiWithoutTokenizerBlock);
    RegisterTest("loader_rejects_invalid_tokenizer_type",
                 TestLoaderRejectsInvalidTokenizerType);
    RegisterTest("loader_rejects_duplicate_tensor_name",
                 TestLoaderRejectsDuplicateTensorName);
    RegisterTest("loader_rejects_required_shape_mismatch",
                 TestLoaderRejectsRequiredShapeMismatch);
    RegisterTest("loader_rejects_overlapping_tensor_ranges",
                 TestLoaderRejectsOverlappingTensorRanges);
    RegisterTest("loader_rejects_missing_dtype", TestLoaderRejectsMissingDtype);
    RegisterTest("loader_rejects_fractional_integer_config",
                 TestLoaderRejectsFractionalIntegerConfig);
    RegisterTest("loader_rejects_nonfinite_float_config",
                 TestLoaderRejectsNonfiniteFloatConfig);
    RegisterTest("loader_rejects_trailing_weight_bytes",
                 TestLoaderRejectsTrailingWeightBytes);
    RegisterTest("loader_loads_tiny_model", TestLoaderLoadsTinyModel);
  }
} loader_test_registrar;
