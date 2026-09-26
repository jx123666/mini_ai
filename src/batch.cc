#include "mini_llama/batch.h"

namespace mini_llama {

//只有1个token的batch
MiniBatch MiniBatch::Single(int token, int pos) {
  MiniBatch batch;
  batch.tokens.push_back(token);
  batch.positions.push_back(pos);
  return batch;
}

//一串token的batch，位置连续递增
MiniBatch MiniBatch::FromTokens(const std::vector<int>& toks, int start_pos) {
  MiniBatch batch;
  batch.tokens = toks;
  batch.positions.reserve(toks.size());
  //第i个token的位置=start_pos+i
  for (size_t i = 0; i < toks.size(); ++i) {
    batch.positions.push_back(start_pos + static_cast<int>(i));
  }
  return batch;
}

}
