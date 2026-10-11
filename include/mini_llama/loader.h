//模型加载：读 model.json 清单 + model.bin 权重
//还能 inspect 只看元数据，不把权重读进内存
#ifndef INCLUDE_MINI_LLAMA_LOADER_H_
#define INCLUDE_MINI_LLAMA_LOADER_H_

#include <string>
#include <vector>

#include "mini_llama/model.h"

namespace mini_llama {

//清单里一条 tensor 的描述：名字、形状、类型、在 bin 里的位置
struct TensorInfo {
  std::string name;
  std::vector<int> shape;
  std::string dtype;
  size_t offset = 0;
  size_t byte_size = 0;
};

//清单里可选的分词器信息；没有这块时默认 ascii
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

//解析 model.json；字段不对或 tensor 对不上会抛异常
ModelManifest ParseManifest(const std::string& config_path);

//按清单从 bin 里把权重读出来
MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path);

//只解析清单并打印，不加载权重
bool InspectModel(const std::string& config_path);

}

#endif
