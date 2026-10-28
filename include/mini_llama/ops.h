// CPU 浮点算子，输入形状不匹配时抛出异常。

#ifndef INCLUDE_MINI_LLAMA_OPS_H_
#define INCLUDE_MINI_LLAMA_OPS_H_

#include "mini_llama/tensor.h"
#include "mini_llama/model.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// Matmul: C = A * B
//   A: [m, k],  B: [k, n],  C: [m, n]
//   C[i,j] = sum_t A[i,t] * B[t,j]
// ---------------------------------------------------------------------------
Tensor Matmul(const Tensor& a, const Tensor& b);

// ---------------------------------------------------------------------------
// Linear: y = x * weight^T
//   x:      [in] 或 [1, in]
//   weight: [out, in]   （每一行是一个输出通道的权重）
//   结果秩与 x 相同
//
// 推理里几乎所有「投影层」都是 Linear，而不是直接调用 Matmul。
// ---------------------------------------------------------------------------
Tensor Linear(const Tensor& x, const Tensor& weight);

// ---------------------------------------------------------------------------
// RMSNorm: y_i = x_i / sqrt(mean(x^2) + eps) * weight_i
//   x, weight: [dim]
//   作用：稳定每层激活的尺度，防止数值爆炸/消失
// ---------------------------------------------------------------------------
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

// ---------------------------------------------------------------------------
// Softmax: y_i = exp(x_i - max) / sum_j exp(x_j - max)
//   减 max 是为了数值稳定，避免 exp 溢出
//   输出非负且和为 1，可当作概率
// ---------------------------------------------------------------------------
Tensor Softmax(const Tensor& x);

// SiLU(x) = x * sigmoid(x)，FFN 里常用
Tensor Silu(const Tensor& x);

// 逐元素相乘
Tensor ElementwiseMul(const Tensor& a, const Tensor& b);

// 逐元素相加，残差连接会用到
Tensor ElementwiseAdd(const Tensor& a, const Tensor& b);

// SwiGLU(gate, up) = SiLU(gate) * up，LLaMA FFN 的门控形式
Tensor SwiGlu(const Tensor& gate, const Tensor& up);

// RoPE：给 Q/K 加上位置信息
// q: [n_heads, head_dim]，k: [n_kv_heads, head_dim]，原地修改
void Rope(Tensor& q, Tensor& k, int pos, float theta,
          RopeType type = RopeType::kNormal);

// 返回最大值的下标（greedy 采样常用）
int ArgMax(const Tensor& x);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_OPS_H_
