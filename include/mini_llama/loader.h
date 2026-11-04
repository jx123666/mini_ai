//模型加载：读model.json+model.bin权重
#ifndef INCLUDE_MINI_LLAMA_LOADER_H_
#define INCLUDE_MINI_LLAMA_LOADER_H_

#include <string>
#include <vector>

#include "mini_llama/model.h"

namespace mini_llama {

//一条信息：名字、形状、类型、在bin里的位置
struct TensorInfo {
  std::string name;
  std::vector<int> shape;
  std::string dtype;
  size_t offset = 0;
  size_t byte_size = 0;
};

//可选的分词器信息
struct TokenizerInfo {
  std::string type = "ascii";
  std::string path;
  int bos_id = 1;
  int eos_id = 2;
  int unk_id = 0;
};

struct ModelManifest {
  ModelConfig config;
  TokenizerInfo tokenizer;
  std::vector<TensorInfo> tensors;
};

//解析model.json；字段不对或tensor对不上会抛出异常
ModelManifest ParseManifest(const std::string& config_path);

//从bin里把权重读出来
MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path);

//只解析清单并打印，不加载权重
bool InspectModel(const std::string& config_path);

}

#endif
