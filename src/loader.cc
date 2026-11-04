//JSON+BIN加载器：解析model.json，校验tensor，再按offset读model.bin
//json可以理解成说明书，bin是零件箱；检查不过就不装机
#include "mini_llama/loader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mini_llama {
namespace {

//加载失败时返回空模型，把原因写进load_error
MiniLlamaModel LoadFailure(const std::string& message) {
  MiniLlamaModel model;
  model.loaded = false;
  model.load_error = message;
  std::cerr << "Failed to load model: " << message << std::endl;
  return model;
}

//数字值后面合法的结束符：空白或 , } ]
//不然stoi("4.5")会当成4，假整数就混过去了
bool IsJsonValueTerminator(char c) {
  return std::isspace(static_cast<unsigned char>(c)) || c == ',' || c == '}' ||
         c == ']';
}

//检查parsed个字符之后是不是合法结束，挡住4.5、NaN后面粘别的东西
void ValidateNumberTerminator(const std::string& content, size_t value_pos,
                              size_t parsed, const std::string& key) {
  size_t pos = value_pos + parsed;
  while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  if (pos < content.size() && !IsJsonValueTerminator(content[pos])) {
    throw std::runtime_error("invalid numeric value for: " + key);
  }
}

//在json里找"key":后面的值起始位置，找不到就抛
size_t FindValuePos(const std::string& content, const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("missing key: " + key);
  }
  //找冒号
  size_t colon = content.find(':', key_pos + marker.size());
  if (colon == std::string::npos) {
    throw std::runtime_error("missing ':' after " + key);
  }
  //跳过空白，落到真正的值上
  size_t pos = colon + 1;
  while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  if (pos >= content.size()) {
    throw std::runtime_error("missing value for " + key);
  }
  return pos;
}

//可选字段：没有key就返回npos，有key但没值还是报错
size_t FindOptionalValuePos(const std::string& content,
                            const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker);
  if (key_pos == std::string::npos) {
    return std::string::npos;
  }
  size_t colon = content.find(':', key_pos + marker.size());
  if (colon == std::string::npos) {
    throw std::runtime_error("missing ':' after " + key);
  }
  size_t pos = colon + 1;
  while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  if (pos >= content.size()) {
    throw std::runtime_error("missing value for " + key);
  }
  return pos;
}

//解析必填整数，顺带检查后面不能再有小数点
int ParseIntKey(const std::string& content, const std::string& key) {
  size_t pos = FindValuePos(content, key);
  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(content.substr(pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer for " + key);
  }
  if (parsed == 0) {
    throw std::runtime_error("invalid integer for " + key);
  }
  ValidateNumberTerminator(content, pos, parsed, key);
  return value;
}

//解析必填浮点数
float ParseFloatKey(const std::string& content, const std::string& key) {
  size_t pos = FindValuePos(content, key);
  size_t parsed = 0;
  float value = 0.0f;
  try {
    value = std::stof(content.substr(pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid float for " + key);
  }
  if (parsed == 0) {
    throw std::runtime_error("invalid float for " + key);
  }
  ValidateNumberTerminator(content, pos, parsed, key);
  return value;
}

//可选字符串，没有就用fallback
std::string ParseOptionalString(const std::string& content,
                                const std::string& key,
                                const std::string& fallback) {
  size_t pos = FindOptionalValuePos(content, key);
  if (pos == std::string::npos) {
    return fallback;
  }
  if (content[pos] != '"') {
    throw std::runtime_error("expected string for " + key);
  }
  size_t end = content.find('"', pos + 1);
  if (end == std::string::npos) {
    throw std::runtime_error("unterminated string for " + key);
  }
  return content.substr(pos + 1, end - pos - 1);
}

//可选整数，没有就用fallback
int ParseOptionalInt(const std::string& content, const std::string& key,
                     int fallback) {
  size_t pos = FindOptionalValuePos(content, key);
  if (pos == std::string::npos) {
    return fallback;
  }
  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(content.substr(pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer for " + key);
  }
  if (parsed == 0) {
    throw std::runtime_error("invalid integer for " + key);
  }
  ValidateNumberTerminator(content, pos, parsed, key);
  return value;
}

//取出"key":{...}整段对象；没有这块就返回空串
//用depth匹配括号，避免嵌套时截早了
std::string ParseObjectForKey(const std::string& content,
                              const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker);
  if (key_pos == std::string::npos) {
    return "";
  }
  size_t obj_start = content.find('{', key_pos + marker.size());
  if (obj_start == std::string::npos) {
    throw std::runtime_error("missing object for " + key);
  }
  int depth = 0;
  for (size_t pos = obj_start; pos < content.size(); ++pos) {
    if (content[pos] == '{') {
      ++depth;
    } else if (content[pos] == '}') {
      --depth;
      if (depth == 0) {
        return content.substr(obj_start, pos - obj_start + 1);
      }
    }
  }
  throw std::runtime_error("unclosed object for " + key);
}

//从from往后找字符串字段，给tensors数组里单个对象用
std::string ParseStringKeyNear(const std::string& content, size_t from,
                               const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker, from);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("missing string key: " + key);
  }
  size_t colon = content.find(':', key_pos + marker.size());
  size_t q1 = content.find('"', colon + 1);
  size_t q2 = content.find('"', q1 + 1);
  if (q1 == std::string::npos || q2 == std::string::npos) {
    throw std::runtime_error("invalid string for " + key);
  }
  return content.substr(q1 + 1, q2 - q1 - 1);
}

//解析shape数组，比如[128, 32]
std::vector<int> ParseShapeNear(const std::string& content, size_t from) {
  size_t key_pos = content.find("\"shape\"", from);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("missing shape");
  }
  size_t lb = content.find('[', key_pos);
  size_t rb = content.find(']', lb);
  if (lb == std::string::npos || rb == std::string::npos) {
    throw std::runtime_error("invalid shape array");
  }
  //把括号里的内容按逗号切开
  std::string inner = content.substr(lb + 1, rb - lb - 1);
  std::vector<int> shape;
  std::stringstream ss(inner);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t start = item.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
      continue;//空段跳过
    }
    size_t end = item.find_last_not_of(" \t\n\r");
    shape.push_back(std::stoi(item.substr(start, end - start + 1)));
  }
  return shape;
}

//解析offset / byte_size这类非负整数
size_t ParseSizeKeyNear(const std::string& content, size_t from,
                        const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker, from);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("missing " + key);
  }
  size_t colon = content.find(':', key_pos + marker.size());
  size_t pos = colon + 1;
  while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  size_t parsed = 0;
  uint64_t value = 0;
  try {
    value = std::stoull(content.substr(pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid " + key);
  }
  if (parsed == 0) {
    throw std::runtime_error("invalid " + key);
  }
  ValidateNumberTerminator(content, pos, parsed, key);
  return static_cast<size_t>(value);
}

//config本身也要合理，否则后面推期望shape会出错
void ValidateConfig(const ModelConfig& config) {
  if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
      config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
      config.max_seq_len <= 0) {
    throw std::runtime_error("all model dimensions must be positive");
  }
  //多头注意力要求dim能整除n_heads
  if (config.dim % config.n_heads != 0) {
    throw std::runtime_error("dim must be divisible by n_heads");
  }
  //GQA：若干个query头共用一个kv头，所以n_heads要能整除n_kv_heads
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::runtime_error("n_heads must be divisible by n_kv_heads");
  }
  //RoPE按复数对处理，head_dim得是偶数
  if (config.head_dim <= 0 || config.head_dim % 2 != 0) {
    throw std::runtime_error("head_dim must be a positive even number");
  }
  if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
    throw std::runtime_error("rope_theta must be positive");
  }
  if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0f) {
    throw std::runtime_error("rms_norm_eps must be positive");
  }
}

//按shape算出float32该占多少字节，也防止乘爆size_t
size_t CheckedTensorByteSize(const TensorInfo& info) {
  size_t num_elements = 1;
  for (int dim : info.shape) {
    if (dim <= 0) {
      throw std::runtime_error("invalid tensor shape for " + info.name);
    }
    const size_t d = static_cast<size_t>(dim);
    //先检查再乘，避免溢出绕回一个小数
    if (num_elements > std::numeric_limits<size_t>::max() / d) {
      throw std::runtime_error("tensor size overflow for " + info.name);
    }
    num_elements *= d;
  }
  if (num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error("tensor byte size overflow for " + info.name);
  }
  return num_elements * sizeof(float);
}

//根据config推出每张必填权重该长什么样
//清单里的shape必须跟这里对得上
std::vector<std::pair<std::string, std::vector<int>>> ExpectedTensorShapes(
    const ModelConfig& config) {
  std::vector<std::pair<std::string, std::vector<int>>> expected = {
      {"token_embedding", {config.vocab_size, config.dim}},
      {"final_norm", {config.dim}},
      {"lm_head", {config.vocab_size, config.dim}},
  };
  for (int layer = 0; layer < config.n_layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    expected.push_back({prefix + "attention_norm", {config.dim}});
    //wq/wo按query头数，wk/wv按kv头数
    expected.push_back(
        {prefix + "wq", {config.n_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wk", {config.n_kv_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wv", {config.n_kv_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wo", {config.dim, config.n_heads * config.head_dim}});
    expected.push_back({prefix + "ffn_norm", {config.dim}});
    expected.push_back({prefix + "w_gate", {config.hidden_dim, config.dim}});
    expected.push_back({prefix + "w_up", {config.hidden_dim, config.dim}});
    expected.push_back({prefix + "w_down", {config.dim, config.hidden_dim}});
  }
  return expected;
}

//shape打成[128, 32]这种，报错好看一点
std::string ShapeToString(const std::vector<int>& shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      result += ", ";
    }
    result += std::to_string(shape[i]);
  }
  result += "]";
  return result;
}

//解析可选的tokenizer块；没有就保持ascii默认
TokenizerInfo ParseTokenizerJson(const std::string& content) {
  TokenizerInfo tokenizer;
  std::string obj = ParseObjectForKey(content, "tokenizer");
  if (obj.empty()) {
    return tokenizer;//没有tokenizer块就用ascii默认
  }

  tokenizer.type = ParseOptionalString(obj, "type", tokenizer.type);
  tokenizer.path = ParseOptionalString(obj, "path", tokenizer.path);
  tokenizer.bos_id = ParseOptionalInt(obj, "bos_id", tokenizer.bos_id);
  tokenizer.eos_id = ParseOptionalInt(obj, "eos_id", tokenizer.eos_id);
  tokenizer.unk_id = ParseOptionalInt(obj, "unk_id", tokenizer.unk_id);

  //只支持ascii和json_vocab
  if (tokenizer.type != "ascii" && tokenizer.type != "json_vocab") {
    throw std::runtime_error("unsupported tokenizer type: " + tokenizer.type);
  }
  if (tokenizer.bos_id < 0 || tokenizer.eos_id < 0 || tokenizer.unk_id < 0) {
    throw std::runtime_error("tokenizer special ids must be non-negative");
  }
  //三个特殊id不能相同
  if (tokenizer.bos_id == tokenizer.eos_id ||
      tokenizer.bos_id == tokenizer.unk_id ||
      tokenizer.eos_id == tokenizer.unk_id) {
    throw std::runtime_error("tokenizer special ids must be distinct");
  }
  if (tokenizer.type == "json_vocab" && tokenizer.path.empty()) {
    throw std::runtime_error("json_vocab tokenizer requires a path");
  }
  return tokenizer;
}


void ValidateManifest(const ModelManifest& manifest) {
  if (manifest.tensors.empty()) {
    throw std::runtime_error("model.json contains no tensors");
  }

  std::map<std::string, TensorInfo> tensor_map;
  std::vector<TensorInfo> by_offset;
  by_offset.reserve(manifest.tensors.size());

  for (const auto& info : manifest.tensors) {
    //当前只支持float32
    if (info.dtype != "float32") {
      throw std::runtime_error("unsupported dtype '" + info.dtype +
                               "' for tensor: " + info.name);
    }
    if (info.shape.empty()) {
      throw std::runtime_error("tensor shape must not be empty: " + info.name);
    }
    //必须等于shape推出来的
    size_t expected_bytes = CheckedTensorByteSize(info);
    if (expected_bytes != info.byte_size) {
      throw std::runtime_error("tensor byte_size mismatch for " + info.name +
                               ": expected " + std::to_string(expected_bytes) +
                               ", manifest says " +
                               std::to_string(info.byte_size));
    }
    if (info.offset > std::numeric_limits<size_t>::max() - info.byte_size) {
      throw std::runtime_error("tensor offset overflow for " + info.name);
    }
    //重名会盖掉另一块权重，直接拒绝
    auto inserted = tensor_map.emplace(info.name, info);
    if (!inserted.second) {
      throw std::runtime_error("duplicate tensor name: " + info.name);
    }
    by_offset.push_back(info);
  }

  //按offset排序，检查相邻区间有没有重叠
  std::sort(by_offset.begin(), by_offset.end(),
            [](const TensorInfo& a, const TensorInfo& b) {
              return a.offset < b.offset;
            });
  for (size_t i = 1; i < by_offset.size(); ++i) {
    size_t previous_end = by_offset[i - 1].offset + by_offset[i - 1].byte_size;
    if (by_offset[i].offset < previous_end) {
      throw std::runtime_error(
          "tensor ranges overlap: " + by_offset[i - 1].name + " and " +
          by_offset[i].name);
    }
  }

  //必填权重名字和形状必须对得上config
  for (const auto& expected : ExpectedTensorShapes(manifest.config)) {
    auto it = tensor_map.find(expected.first);
    if (it == tensor_map.end()) {
      throw std::runtime_error("missing required tensor: " + expected.first);
    }
    if (it->second.shape != expected.second) {
      throw std::runtime_error("tensor shape mismatch for " + expected.first +
                               ": expected " + ShapeToString(expected.second) +
                               ", manifest says " +
                               ShapeToString(it->second.shape));
    }
  }
}

//从bin里按offset读一块float32
Tensor LoadTensorAt(std::ifstream& wf, const TensorInfo& info) {
  Tensor t(info.shape, 0.0f);
  wf.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
  if (!wf.good()) {
    throw std::runtime_error("failed to seek tensor: " + info.name);
  }
  wf.read(reinterpret_cast<char*>(t.data.data()),
          static_cast<std::streamsize>(info.byte_size));
  //实际读到的字节数必须刚好
  if (static_cast<size_t>(wf.gcount()) != info.byte_size) {
    throw std::runtime_error("failed to read tensor: " + info.name);
  }
  return t;
}

//按名字从map里找到TensorInfo再读
Tensor LoadByName(std::ifstream& wf,
                  const std::map<std::string, TensorInfo>& map,
                  const std::string& name) {
  auto it = map.find(name);
  if (it == map.end()) {
    throw std::runtime_error("missing tensor: " + name);
  }
  return LoadTensorAt(wf, it->second);
}

}

//解析model.json：config + tokenizer + tensors，最后做整表校验
ModelManifest ParseManifest(const std::string& config_path) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open config: " + config_path);
  }
  //整文件读进内存，后面用字符串查找凑合解析
  std::stringstream buffer;
  buffer << file.rdbuf();
  const std::string content = buffer.str();

  ModelManifest manifest;
  ModelConfig& c = manifest.config;
  c.vocab_size = ParseIntKey(content, "vocab_size");
  c.dim = ParseIntKey(content, "dim");
  c.hidden_dim = ParseIntKey(content, "hidden_dim");
  c.n_layers = ParseIntKey(content, "n_layers");
  c.n_heads = ParseIntKey(content, "n_heads");
  c.n_kv_heads = ParseIntKey(content, "n_kv_heads");
  c.max_seq_len = ParseIntKey(content, "max_seq_len");
  c.rope_theta = ParseFloatKey(content, "rope_theta");
  c.rms_norm_eps = ParseFloatKey(content, "rms_norm_eps");
  c.head_dim = c.n_heads > 0 ? c.dim / c.n_heads : 0;
  const std::string rope = ParseOptionalString(content, "rope_type", "normal");
  if (rope != "normal" && rope != "neox") {
    throw std::runtime_error("unsupported rope_type: " + rope);
  }
  c.rope_type = rope == "neox" ? RopeType::kNeoX : RopeType::kNormal;
  ValidateConfig(c);

  manifest.tokenizer = ParseTokenizerJson(content);

  //逐个解析tensors数组里的对象
  size_t tensors_pos = content.find("\"tensors\"");
  if (tensors_pos == std::string::npos) {
    throw std::runtime_error("missing tensors array");
  }
  size_t arr = content.find('[', tensors_pos);
  size_t pos = arr + 1;
  while (pos < content.size()) {
    size_t obj = content.find('{', pos);
    if (obj == std::string::npos) {
      break;
    }
    size_t end = content.find('}', obj);
    if (end == std::string::npos) {
      break;
    }

    //字段必须落在当前{}里，避免吃到下一个对象的值
    auto in_object = [&](const std::string& key) {
      size_t key_pos = content.find("\"" + key + "\"", obj);
      return key_pos != std::string::npos && key_pos < end;
    };

    //先拿name，后面缺字段时报错能带上名字
    std::string maybe_name = in_object("name")
                                 ? ParseStringKeyNear(content, obj, "name")
                                 : "<unknown>";
    if (!in_object("name") || maybe_name.empty()) {
      throw std::runtime_error("tensor missing name field");
    }
    if (!in_object("dtype")) {
      throw std::runtime_error("tensor missing dtype field: " + maybe_name);
    }
    if (!in_object("shape")) {
      throw std::runtime_error("tensor missing shape field: " + maybe_name);
    }
    if (!in_object("offset")) {
      throw std::runtime_error("tensor missing offset field: " + maybe_name);
    }
    if (!in_object("byte_size")) {
      throw std::runtime_error("tensor missing byte_size field: " + maybe_name);
    }

    TensorInfo info;
    info.name = maybe_name;
    info.shape = ParseShapeNear(content, obj);
    info.dtype = ParseStringKeyNear(content, obj, "dtype");
    info.offset = ParseSizeKeyNear(content, obj, "offset");
    info.byte_size = ParseSizeKeyNear(content, obj, "byte_size");
    manifest.tensors.push_back(info);

    pos = end + 1;
    //下一个{在]后面，说明数组结束了
    size_t following_obj = content.find('{', pos);
    size_t arr_end = content.find(']', pos);
    if (arr_end != std::string::npos &&
        (following_obj == std::string::npos || arr_end < following_obj)) {
      break;
    }
  }

  ValidateManifest(manifest);
  return manifest;
}

//按清单把bin里的权重量进MiniLlamaModel
MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path) {
  ModelManifest manifest;
  try {
    manifest = ParseManifest(config_path);
  } catch (const std::exception& e) {
    return LoadFailure(e.what());
  }

  //名字 -> TensorInfo，后面按名取
  std::map<std::string, TensorInfo> tensor_map;
  for (const auto& info : manifest.tensors) {
    tensor_map[info.name] = info;
  }

  std::ifstream wf(weights_path, std::ios::binary);
  if (!wf.is_open()) {
    return LoadFailure("failed to open weights: " + weights_path);
  }

  MiniLlamaModel model;
  model.config = manifest.config;
  try {
    //先读embedding，再按层读attn/ffn，最后final_norm和lm_head
    model.token_embedding = LoadByName(wf, tensor_map, "token_embedding");
    model.layers.resize(manifest.config.n_layers);
    for (int layer = 0; layer < manifest.config.n_layers; ++layer) {
      LayerWeights& lw = model.layers[layer];
      const std::string prefix = "layers." + std::to_string(layer) + ".";
      lw.attention_norm = LoadByName(wf, tensor_map, prefix + "attention_norm");
      lw.wq = LoadByName(wf, tensor_map, prefix + "wq");
      lw.wk = LoadByName(wf, tensor_map, prefix + "wk");
      lw.wv = LoadByName(wf, tensor_map, prefix + "wv");
      if (tensor_map.count(prefix + "bq")) lw.bq = LoadByName(wf, tensor_map, prefix + "bq");
      if (tensor_map.count(prefix + "bk")) lw.bk = LoadByName(wf, tensor_map, prefix + "bk");
      if (tensor_map.count(prefix + "bv")) lw.bv = LoadByName(wf, tensor_map, prefix + "bv");
      if (!lw.bq.data.empty()) lw.bq.AssertShape({model.config.dim}, "bq");
      const int kv_dim = model.config.n_kv_heads * model.config.head_dim;
      if (!lw.bk.data.empty()) lw.bk.AssertShape({kv_dim}, "bk");
      if (!lw.bv.data.empty()) lw.bv.AssertShape({kv_dim}, "bv");
      lw.wo = LoadByName(wf, tensor_map, prefix + "wo");
      lw.ffn_norm = LoadByName(wf, tensor_map, prefix + "ffn_norm");
      lw.w_gate = LoadByName(wf, tensor_map, prefix + "w_gate");
      lw.w_up = LoadByName(wf, tensor_map, prefix + "w_up");
      lw.w_down = LoadByName(wf, tensor_map, prefix + "w_down");
    }
    model.final_norm = LoadByName(wf, tensor_map, "final_norm");
    model.lm_head = LoadByName(wf, tensor_map, "lm_head");
  } catch (const std::exception& e) {
    return LoadFailure(e.what());
  }

  //bin不能多也不能少，末尾多出来的字节通常是写错
  wf.clear();
  wf.seekg(0, std::ios::end);
  std::streamoff file_size = wf.tellg();
  if (file_size < 0) {
    return LoadFailure("failed to determine weights file size");
  }
  //期望长度，最靠后那块tensor的结尾
  size_t expected_end = 0;
  for (const auto& info : manifest.tensors) {
    size_t tensor_end = info.offset + info.byte_size;
    if (tensor_end > expected_end) {
      expected_end = tensor_end;
    }
  }
  if (static_cast<size_t>(file_size) != expected_end) {
    if (static_cast<size_t>(file_size) > expected_end) {
      return LoadFailure("weights file has trailing bytes: file size=" +
                         std::to_string(file_size) +
                         ", expected=" + std::to_string(expected_end));
    }
    return LoadFailure(
        "weights file is too short: file size=" + std::to_string(file_size) +
        ", expected=" + std::to_string(expected_end));
  }

  model.loaded = true;
  return model;
}

//只解析清单并打印，不读bin，方便检查模型文件写对没有
bool InspectModel(const std::string& config_path) {
  ModelManifest manifest;
  try {
    manifest = ParseManifest(config_path);
  } catch (const std::exception& e) {
    std::cerr << "Failed to inspect model: " << e.what() << std::endl;
    return false;
  }

  //先打config
  const ModelConfig& c = manifest.config;
  std::cout << "=== Model Config ===" << std::endl;
  std::cout << "  vocab_size:    " << c.vocab_size << std::endl;
  std::cout << "  dim:           " << c.dim << std::endl;
  std::cout << "  hidden_dim:    " << c.hidden_dim << std::endl;
  std::cout << "  n_layers:      " << c.n_layers << std::endl;
  std::cout << "  n_heads:       " << c.n_heads << std::endl;
  std::cout << "  n_kv_heads:    " << c.n_kv_heads << std::endl;
  std::cout << "  head_dim:      " << c.head_dim << std::endl;
  std::cout << "  max_seq_len:   " << c.max_seq_len << std::endl;
  std::cout << "  rope_theta:    " << c.rope_theta << std::endl;
  std::cout << "  rms_norm_eps:  " << c.rms_norm_eps << std::endl;

  //再打每张tensor的快递单
  std::cout << std::endl;
  std::cout << "=== Tensors (" << manifest.tensors.size() << ") ==="
            << std::endl;
  size_t total_bytes = 0;
  for (const auto& t : manifest.tensors) {
    std::cout << "  " << t.name << std::endl;
    std::cout << "    shape:     " << ShapeToString(t.shape) << std::endl;
    std::cout << "    dtype:     " << t.dtype << std::endl;
    std::cout << "    offset:    " << t.offset << std::endl;
    std::cout << "    byte_size: " << t.byte_size << std::endl;
    total_bytes += t.byte_size;
  }

  //最后打tokenizer信息
  std::cout << std::endl;
  std::cout << "=== Tokenizer ===" << std::endl;
  std::cout << "  type: " << manifest.tokenizer.type << std::endl;
  std::cout << "  path: "
            << (manifest.tokenizer.path.empty() ? "<none>"
                                                : manifest.tokenizer.path)
            << std::endl;
  std::cout << "  bos_id: " << manifest.tokenizer.bos_id << std::endl;
  std::cout << "  eos_id: " << manifest.tokenizer.eos_id << std::endl;
  std::cout << "  unk_id: " << manifest.tokenizer.unk_id << std::endl;

  std::cout << std::endl;
  std::cout << "Total weight bytes: " << total_bytes << std::endl;
  return true;
}

}
