#include "mini_llama/loader.h"
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {
namespace {

MiniLlamaModel LoadFailure(const std::string& message) {
  MiniLlamaModel model;
  model.loaded = false;
  model.load_error = message;
  std::cerr << "Failed to load model: " << message << std::endl;
  return model;
}

//在json里找key:后面的数字起始位置
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
  //找数字起始位置
  size_t pos = colon + 1;
  while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  return pos;
}

//解析整数
int ParseIntKey(const std::string& content, const std::string& key) {
  size_t pos = FindValuePos(content, key);
  size_t parsed = 0;
  return std::stoi(content.substr(pos), &parsed);
}

//解析浮点数
float ParseFloatKey(const std::string& content, const std::string& key) {
  size_t pos = FindValuePos(content, key);
  size_t parsed = 0;
  return std::stof(content.substr(pos), &parsed);
}

//解析 name: xxx
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

//解析shape
std::vector<int> ParseShapeNear(const std::string& content, size_t from) {
  size_t key_pos = content.find("\"shape\"", from);
  //找左括号
  size_t lb = content.find('[', key_pos);
  //找右括号
  size_t rb = content.find(']', lb);
  if (rb == std::string::npos) {
    throw std::runtime_error("missing ']' after shape");
  }
  //解析内部内容
  std::string inner = content.substr(lb + 1, rb - lb - 1);
  std::vector<int> shape;
  std::stringstream ss(inner);
  std::string item;
  while (std::getline(ss, item, ',')) {
    //找第一个非空字符
    size_t start = item.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
      continue;
    }
    shape.push_back(std::stoi(item.substr(start)));
  }
  return shape;
}
//解析size
size_t ParseSizeKeyNear(const std::string& content, size_t from,
                        const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker, from);
  size_t colon = content.find(':', key_pos + marker.size());
  size_t pos = colon + 1;
  while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  size_t parsed = 0;
  return static_cast<size_t>(std::stoull(content.substr(pos), &parsed));
}

//从bin里按offset读一块float32
Tensor LoadTensorAt(std::ifstream& wf, const TensorInfo& info) {
  size_t numel = 1;
  for (int d : info.shape) {
    numel *= static_cast<size_t>(d);
  }
  if (info.byte_size != numel * sizeof(float)) {
    throw std::runtime_error("tensor byte_size mismatch: " + info.name);
  }
  Tensor t(info.shape, 0.0f);
  wf.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
  wf.read(reinterpret_cast<char*>(t.data.data()),
          static_cast<std::streamsize>(info.byte_size));
  if (!wf) {
    throw std::runtime_error("failed to read tensor: " + info.name);
  }
  return t;
}

//按name从bin里读tensor
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

//解析模型清单
ModelManifest ParseManifest(const std::string& config_path) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open config: " + config_path);
  }
  //读文件内容
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
  c.head_dim = c.dim / c.n_heads;//tiny 模型没单独写 head_dim

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
    TensorInfo info;
    info.name = ParseStringKeyNear(content, obj, "name");
    info.shape = ParseShapeNear(content, obj);
    info.dtype = ParseStringKeyNear(content, obj, "dtype");
    info.offset = ParseSizeKeyNear(content, obj, "offset");
    info.byte_size = ParseSizeKeyNear(content, obj, "byte_size");
    manifest.tensors.push_back(info);
    pos = end + 1;
    //数组结束后停
    size_t next_obj = content.find('{', pos);
    size_t arr_end = content.find(']', pos);
    if (arr_end != std::string::npos &&
        (next_obj == std::string::npos || arr_end < next_obj)) {
      break;
    }
  }
  return manifest;
}

//加载模型权重
MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path) {
  ModelManifest manifest;
  //解析模型清单
  try {
    manifest = ParseManifest(config_path);
  } catch (const std::exception& e) {
    return LoadFailure(e.what());
  }

  std::map<std::string, TensorInfo> tensor_map;
  for (const auto& info : manifest.tensors) {
    tensor_map[info.name] = info;
  }

  //打开权重文件
  std::ifstream wf(weights_path, std::ios::binary);
  if (!wf.is_open()) {
    return LoadFailure("failed to open weights: " + weights_path);
  }
//填充模型权重
  MiniLlamaModel model;
  model.config = manifest.config;
  try {
    model.token_embedding = LoadByName(wf, tensor_map, "token_embedding");
    model.layers.resize(manifest.config.n_layers);
    for (int layer = 0; layer < manifest.config.n_layers; ++layer) {
      LayerWeights& lw = model.layers[layer];
      const std::string prefix = "layers." + std::to_string(layer) + ".";
      lw.attention_norm = LoadByName(wf, tensor_map, prefix + "attention_norm");
      lw.wq = LoadByName(wf, tensor_map, prefix + "wq");
      lw.wk = LoadByName(wf, tensor_map, prefix + "wk");
      lw.wv = LoadByName(wf, tensor_map, prefix + "wv");
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

  model.loaded = true;
  return model;
}

}
