#ifndef INCLUDE_MINI_LLAMA_CONTEXT_H_
#define INCLUDE_MINI_LLAMA_CONTEXT_H_

#include <vector>
#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"

namespace mini_llama {

struct MiniLlamaContext {
  const MiniLlamaModel* model = nullptr;
  KvCache kv_cache;
  int pos = 0;//当前算到第几个位置

  //喂进去过的全部token
  std::vector<int> token_history;

  //记录prefill，decode各跑了多少个
  int n_prefill_tokens = 0;
  int n_decode_tokens = 0;

  MiniLlamaContext() = default;
  MiniLlamaContext(const MiniLlamaContext&) = delete;
  MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;
  MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
  MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;

  //按model.config建空白KV Cache
  explicit MiniLlamaContext(const MiniLlamaModel* model);
};

}

#endif
