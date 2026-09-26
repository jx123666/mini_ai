#include "mini_llama/ops.h"
#include <cmath>
#include <stdexcept>
#include "mini_llama/matmul_dispatch.h"

namespace mini_llama {

Tensor Matmul(const Tensor& a, const Tensor& b) {
  return MatmulDispatch(a, b, DefaultMatmulMode());
}

Tensor Linear(const Tensor& x, const Tensor& weight) {
  return LinearDispatch(x, weight, DefaultMatmulMode());
}

//RMSNorm实现
//输入x的形状为[dim]
//输入weight的形状为[dim]
//输出y的形状为[dim]
//输出y的每个元素为x与weight的点积
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps) {
  if (x.num_dims() != 1 || weight.num_dims() != 1) {
    throw std::runtime_error("RmsNorm: expected 1D tensors");
  }
  //如果x或weight的维度不是1，异常
  const int dim = x.shape[0];
  if (weight.shape[0] != dim) {
    throw std::runtime_error("RmsNorm: dimension mismatch");
  }
  //计算x的平方和
  float ss = 0.0f;
  for (int i = 0; i < dim; ++i) {
    ss += x.data[i] * x.data[i];
  }
  ss /= static_cast<float>(dim);
  const float scale = 1.0f / std::sqrt(ss + eps);
  //计算y的每个元素
  Tensor y({dim}, 0.0f);
  for (int i = 0; i < dim; ++i) {
    y.data[i] = x.data[i] * scale * weight.data[i];
  }
  return y;
}
//Softmax实现
//输入x的形状为n
//输出y的形状为n
//输出y的每个元素为x的指数除以x的指数和
Tensor Softmax(const Tensor& x) {
  if (x.num_dims() != 1) {
    throw std::runtime_error("Softmax: expected 1D tensor, got " +
                             x.ShapeStringShort());
  }
  if (x.size() == 0) {
    throw std::runtime_error("Softmax: cannot compute Softmax of empty tensor");
  }
//获取x的形状
  const int n = x.shape[0];
//获取x的最大值
  float max_val = x.data[0];
  for (int i = 1; i < n; ++i) {
    if (x.data[i] > max_val) {
      max_val = x.data[i];
    }
  }
//计算x的指数和
  float sum = 0.0f;
  Tensor y({n}, 0.0f);
  //计算y的每个元素
  for (int i = 0; i < n; ++i) {
    y.data[i] = std::exp(x.data[i] - max_val);
    sum += y.data[i];
  }

  //归一化，使概率和为 1
  for (int i = 0; i < n; ++i) {
    y.data[i] /= sum;
  }
  return y;
}

//SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//输入x的形状为n
//输出y的形状为n
//输出y的每个元素为x的sigmoid乘以x
Tensor Silu(const Tensor& x) {
  Tensor y(x.shape, 0.0f);
  for (size_t i = 0; i < x.data.size(); ++i) {
    const float val = x.data[i];
    y.data[i] = val / (1.0f + std::exp(-val));
  }
  return y;
}

//ElementwiseMul：逐元素相乘
Tensor ElementwiseMul(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) {
    throw std::runtime_error("ElementwiseMul: shape mismatch " +
                             a.ShapeStringShort() + " vs " +
                             b.ShapeStringShort());
  }
  Tensor y(a.shape, 0.0f);
  for (size_t i = 0; i < a.data.size(); ++i) {
    y.data[i] = a.data[i] * b.data[i];
  }
  return y;
}

//ElementwiseAdd：逐元素相加，残差 x + sublayer(x)
Tensor ElementwiseAdd(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) {
    throw std::runtime_error("ElementwiseAdd: shape mismatch " +
                             a.ShapeStringShort() + " vs " +
                             b.ShapeStringShort());
  }
  Tensor y(a.shape, 0.0f);
  for (size_t i = 0; i < a.data.size(); ++i) {
    y.data[i] = a.data[i] + b.data[i];
  }
  return y;
}

//SwiGlu实现
//输入gate的形状为[n]
//输入up的形状为[n]
//输出y的形状为[n]
//输出y的每个元素为gate的sigmoid乘以up
Tensor SwiGlu(const Tensor& gate, const Tensor& up) {
  if (gate.shape != up.shape) {
    throw std::runtime_error("SwiGlu: shape mismatch " +
                             gate.ShapeStringShort() + " vs " +
                             up.ShapeStringShort());
  }
  Tensor y(gate.shape, 0.0f);

  for (size_t i = 0; i < gate.data.size(); ++i) {
    const float val = gate.data[i];
    const float sigmoid = 1.0f / (1.0f + std::exp(-val));//计算gate的sigmoid
    y.data[i] = val * sigmoid * up.data[i];//计算y的每个元素
  }
  return y;
}

//对每个 head 做标准 RoPE：相邻两维 (x0,x1) 旋转一下
void ApplyRopeNormalToHeads(Tensor& x, int pos, float theta) {
  const int n_heads = x.shape[0];
  const int head_dim = x.shape[1];
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < head_dim; i += 2) {
      const float freq =
          1.0f / std::pow(theta, static_cast<float>(i) /
                                     static_cast<float>(head_dim));
      const float cos_val = std::cos(static_cast<float>(pos) * freq);
      const float sin_val = std::sin(static_cast<float>(pos) * freq);
      const float x0 = x.data[h * head_dim + i];
      const float x1 = x.data[h * head_dim + i + 1];
      x.data[h * head_dim + i] = x0 * cos_val - x1 * sin_val;
      x.data[h * head_dim + i + 1] = x0 * sin_val + x1 * cos_val;
    }
  }
}

//RoPE：给 Q、K 写入位置信息，Attention 才能区分前后
void Rope(Tensor& q, Tensor& k, int pos, float theta) {
  if (q.num_dims() != 2 || k.num_dims() != 2) {
    throw std::runtime_error("Rope: expected 2D tensors");
  }
  if (pos < 0) {
    throw std::out_of_range("Rope: position must be non-negative");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error("Rope: theta must be finite and positive");
  }
  if (q.shape[1] != k.shape[1]) {
    throw std::runtime_error("Rope: q and k head_dim mismatch");
  }
  if (q.shape[1] <= 0 || q.shape[1] % 2 != 0) {
    throw std::runtime_error("Rope: head_dim must be positive and even");
  }
  ApplyRopeNormalToHeads(q, pos, theta);
  ApplyRopeNormalToHeads(k, pos, theta);
}

//ArgMax：找最大值下标，greedy 采样用
int ArgMax(const Tensor& x) {
  if (x.size() == 0) {
    throw std::runtime_error("ArgMax: cannot compute ArgMax of empty tensor");
  }
  int best = 0;
  float best_val = x.data[0];
  for (size_t i = 1; i < x.data.size(); ++i) {//计算x的最大值
    if (x.data[i] > best_val) {
      best_val = x.data[i];
      best = static_cast<int>(i);
    }
  }
  return best;
}

} 
