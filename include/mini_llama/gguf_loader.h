//GGUF MiniLlamaModel：把llama.cpp风格的张量名映射进我们的模型结构
//支持 F32、F16 和 Q8_0 / Q4_0 / Q4_1；量化线性层保留压缩权重。
#ifndef INCLUDE_MINI_LLAMA_GGUF_LOADER_H_
#define INCLUDE_MINI_LLAMA_GGUF_LOADER_H_

#include <string>

#include "mini_llama/model.h"

namespace mini_llama {

//从.gguf装模型；失败时loaded=false，原因写在load_error
MiniLlamaModel LoadGgufModel(const std::string& gguf_path);

}

#endif
