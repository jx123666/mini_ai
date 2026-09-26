//推理入口：ForwardToken / ForwardBatch
//链路：embed -> 每层(rmsnorm -> qkv -> rope -> attention -> ffn) -> logits
#ifndef INCLUDE_MINI_LLAMA_FORWARD_H_
#define INCLUDE_MINI_LLAMA_FORWARD_H_

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

//处理 1 个 token，返回 logits，形状 [vocab_size]
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token);

//处理一整批，内部循环调 ForwardToken，返回最后一个的 logits
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch);

}

#endif
