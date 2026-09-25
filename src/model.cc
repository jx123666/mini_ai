#include "mini_llama/model.h"

namespace mini_llama {

MiniLlamaModel MakeToyModel(const ModelConfig& config) {
  MiniLlamaModel model;
  model.config = config;
  model.loaded = true;
  return model;
}

} 
