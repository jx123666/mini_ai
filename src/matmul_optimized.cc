#include "mini_llama/matmul_dispatch.h"

#include <stdexcept>
#include <vector>
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include "mini_llama/thread_pool.h"
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace mini_llama {
namespace {
std::mutex pool_mutex;
std::shared_ptr<ThreadPool> pool;
std::atomic<unsigned> requested_threads{0};
std::atomic<MatmulMode> default_mode{MatmulMode::kThreadedSimd};

std::shared_ptr<ThreadPool> GetPool() {
  std::lock_guard<std::mutex> lock(pool_mutex);
  if (!pool) pool = std::make_shared<ThreadPool>(MatmulThreads());
  return pool;
}

float Dot(const float* a, const float* b, int n, bool simd) {
  int i = 0;
  float sum = 0;
#if defined(__SSE2__) || defined(_M_X64)
  if (simd) {
    __m128 acc = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
      acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    float lanes[4];
    _mm_storeu_ps(lanes, acc);
    sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
  }
#else
  (void)simd;
#endif
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

void RunRows(int rows, MatmulMode mode, const std::function<void(int)>& fn) {
  const bool threaded = mode == MatmulMode::kThreaded || mode == MatmulMode::kThreadedSimd;
  if (threaded && rows >= 16 && MatmulThreads() > 1) GetPool()->ParallelFor(rows, fn);
  else for (int row = 0; row < rows; ++row) fn(row);
}
}

unsigned MatmulThreads() {
  const unsigned count = requested_threads.load();
  return count ? count : std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
}

void SetMatmulThreads(unsigned count) {
  if (count > 256) throw std::invalid_argument("thread count exceeds 256");
  std::lock_guard<std::mutex> lock(pool_mutex);
  requested_threads = count;
  pool.reset();
}

void SetDefaultMatmulMode(MatmulMode mode) { default_mode = mode; }

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
  return default_mode.load();
}
//矩阵乘法调度函数
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
  if (a.quantized || b.quantized) throw std::runtime_error("Matmul requires F32 inputs; use Linear for quantized weights");
  //如果a或b的维度不是2，异常
  if (a.num_dims() != 2 || b.num_dims() != 2) {
    throw std::runtime_error("MatmulDispatch: expected 2D tensors");
  }
  //如果a的列数不等于b的行数，异常
  if (a.shape[1] != b.shape[0]) {
    throw std::runtime_error("MatmulDispatch: dimension mismatch");
  }
  //根据mode选择不同的矩阵乘法实现
  if (mode == MatmulMode::kNaive) return MatmulNaive(a, b);
  const int m = a.shape[0], k = a.shape[1], n = b.shape[1];
  Tensor transposed({n, k});
  for (int t = 0; t < k; ++t)
    for (int j = 0; j < n; ++j) transposed[j * k + t] = b[t * n + j];
  Tensor out({m, n});
  const bool simd = mode == MatmulMode::kSimd || mode == MatmulMode::kThreadedSimd;
  RunRows(m, mode, [&](int row) {
    for (int col = 0; col < n; ++col)
      out[row * n + col] = Dot(a.data.data() + row * k,
                               transposed.data.data() + col * k, k, simd);
  });
  return out;
}
//线性层调度函数
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
  if (x.quantized) throw std::runtime_error("Linear requires F32 activations");
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
  if (x.data.size() != static_cast<size_t>(in_features) ||
      (!weight.quantized && weight.data.size() != static_cast<size_t>(weight.shape[0]) * in_features))
    throw std::runtime_error("LinearDispatch: storage size does not match shape");

  const int rows = weight.shape[0];
  Tensor out(x.num_dims() == 1 ? std::vector<int>{rows} : std::vector<int>{1, rows});
  if (weight.quantized) {
    RunRows(rows, mode, [&](int row) { out[row] = QuantizedDotRow(*weight.quantized, row, x.data.data()); });
    return out;
  }
  if (mode == MatmulMode::kNaive) return LinearNaive(x, weight);
  const bool simd = mode == MatmulMode::kSimd || mode == MatmulMode::kThreadedSimd;
  RunRows(rows, mode, [&](int row) {
    out[row] = Dot(x.data.data(), weight.data.data() + row * in_features, in_features, simd);
  });
  return out;
}

}  
