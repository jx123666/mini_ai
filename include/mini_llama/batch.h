#ifndef INCLUDE_MINI_LLAMA_BATCH_H_
#define INCLUDE_MINI_LLAMA_BATCH_H_

#include <vector>
namespace mini_llama {

struct MiniBatch {
  std::vector<int> tokens;//token id
  std::vector<int> positions;//对应位置

  int num_tokens() const { return static_cast<int>(tokens.size()); }

  //Decode 一个 token、一个位置
  static MiniBatch Single(int token, int pos);

  //Prefill 一串token，位置从start_pos连续往上加
  static MiniBatch FromTokens(const std::vector<int>& toks, int start_pos = 0);
};

}

#endif
