#ifndef INCLUDE_MINI_LLAMA_LOADER_H_
#define INCLUDE_MINI_LLAMA_LOADER_H_

#include <string>
#include <vector>
#include "mini_llama/model.h"

namespace mini_llama {
//tensor信息
struct TensorInfo {
  std::string name;
  std::vector<int> shape;
  std::string dtype;
  size_t offset = 0;
  size_t byte_size = 0;
};

struct ModelManifest {
  ModelConfig config;
  std::vector<TensorInfo> tensors;
};

//解析模型清单
ModelManifest ParseManifest(const std::string& config_path);
//加载模型
MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path);

}

#endif
