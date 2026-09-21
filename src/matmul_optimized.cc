#include "mini_llama/matmul_dispatch.h"

#include <stdexcept>
#include <vector>

namespace mini_llama {

//最基础的矩阵乘法实现
//三重循环
//内存下标（row-major）：
//A[i,t]-> a.data[i*k + t]
//B[t,j]-> b.data[t*n + j]
//C[i,j]-> c.data[i*n + j]
static Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
  const int m = a.shape[0];
  const int inner_dim = a.shape[1];  // 即 k
  const int n = b.shape[1];
  Tensor c({m, n}, 0.0f);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int t = 0; t < inner_dim; ++t) {
        sum += a.data[i * inner_dim + t] * b.data[t * n + j];
      }
      c.data[i * n + j] = sum;
    }
  }
  return c;
}

//最基础的线性层实现
//输入x的形状为in_features或[1, in_features]
//输出result的形状为out_features或[1, out_features]
//输出result的每个元素为x与weight的点积
static Tensor LinearNaive(const Tensor& x, const Tensor& weight) {
  const int in_features = weight.shape[1];
  const int out_features = weight.shape[0];
  const bool is_1d = (x.num_dims() == 1);
//根据x的形状，确定result的形状
  Tensor result(is_1d ? std::vector<int>{out_features}
                      : std::vector<int>{1, out_features},
                0.0f);
//计算result的每个元素
  for (int j = 0; j < out_features; ++j) {
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
      sum += x.data[k] * weight.data[j * in_features + k];
    }
    result.data[j] = sum;
  }
  return result;
}

MatmulMode DefaultMatmulMode() {
  //默认使用最基础的矩阵乘法实现
  return MatmulMode::kNaive;
}
//矩阵乘法调度函数
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
  //如果a或b的维度不是2，异常
  if (a.num_dims() != 2 || b.num_dims() != 2) {
    throw std::runtime_error("MatmulDispatch: expected 2D tensors");
  }
  //如果a的列数不等于b的行数，异常
  if (a.shape[1] != b.shape[0]) {
    throw std::runtime_error("MatmulDispatch: dimension mismatch");
  }
  //根据mode选择不同的矩阵乘法实现
  (void)mode;
  return MatmulNaive(a, b);
}
//线性层调度函数
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
  if (weight.num_dims() != 2) {//异常
    throw std::runtime_error(
        "LinearDispatch: expected weight shape [out_features, in_features]");
  }
  //根据x的形状，确定in_features
  int in_features = 0;
  if (x.num_dims() == 1) {//x的形状为[in_features]
    in_features = x.shape[0];
  } else if (x.num_dims() == 2 && x.shape[0] == 1) {//x的形状为[1, in_features]
    in_features = x.shape[1];
  } else {
    throw std::runtime_error(
        "LinearDispatch: expected x shape [in_features] or [1, in_features]");
  }

  if (weight.shape[1] != in_features) {
    throw std::runtime_error("LinearDispatch: dimension mismatch");
  }

  (void)mode;
  return LinearNaive(x, weight);
}

}  
