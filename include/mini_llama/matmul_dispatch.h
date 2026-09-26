// =============================================================================
// 文件作用：matmul_dispatch.h
//   矩阵乘的「分发层」：对外统一接口，对内可切换不同实现。
//
// 为什么要分发？
//   - 学习阶段：用 kNaive 三重循环，最容易手写、最容易用测试验证正确性
//   - 性能阶段：可换成多线程 / SIMD（本教学项目先保留接口，默认走 naive）
//
// 与 mini-llama 课程文档对应：
//   include/mini_llama/matmul_dispatch.h  /  src/matmul_optimized.cc
// =============================================================================

#ifndef INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
#define INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_

#include "mini_llama/tensor.h"

namespace mini_llama {

// 矩阵乘实现模式
enum class MatmulMode {
  kNaive,         // 可读三重循环（学习与验收首选）
  kThreaded,      // 按行并行（进阶）
  kSimd,          // SIMD 点积加速 Linear（进阶）
  kThreadedSimd,  // 线程 + SIMD（进阶）
};

// 推理主路径默认模式（教学项目默认 naive，保证结果可预测、易调试）
MatmulMode DefaultMatmulMode();

// F32 矩阵乘：[m,k] x [k,n] -> [m,n]
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// F32 Linear：x @ weight^T
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
