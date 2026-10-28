// 矩阵乘和线性层的实现选择。

#ifndef INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
#define INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_

#include "mini_llama/tensor.h"

namespace mini_llama {

// 矩阵乘实现模式
enum class MatmulMode {
  kNaive,         // 标量计算
  kThreaded,      // 按输出行并行
  kSimd,          // SIMD 点积
  kThreadedSimd,  // 并行 SIMD 点积
};

// 0 使用自动线程数；1 保持单线程。
void SetMatmulThreads(unsigned count);
unsigned MatmulThreads();
void SetDefaultMatmulMode(MatmulMode mode);
MatmulMode DefaultMatmulMode();

// F32 矩阵乘：[m,k] x [k,n] -> [m,n]
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// F32 Linear：x @ weight^T
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
