#ifndef INCLUDE_MINI_LLAMA_RUNTIME_H_
#define INCLUDE_MINI_LLAMA_RUNTIME_H_

#include "mini_llama/model.h"
#include "mini_llama/tokenizer.h"
#include <memory>
#include <string>

namespace mini_llama {
struct InferenceModel {
  MiniLlamaModel model;
  std::unique_ptr<ITokenizer> tokenizer;
  std::string chat_template;
  std::string source;
};

InferenceModel LoadInferenceModel(const std::string &source,
                                  const std::string &config_path,
                                  const std::string &tokenizer_path = "");
} // namespace mini_llama
#endif
