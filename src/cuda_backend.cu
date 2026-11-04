#include "mini_llama/backend.h"
#include "mini_llama/context.h"

#include <algorithm>
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mini_llama {
namespace {
void Check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

struct DeviceBuffer {
  void *pointer = nullptr;
  int device = 0;
  DeviceBuffer() = default;
  DeviceBuffer(size_t bytes, int id) : device(id) {
    Check(cudaSetDevice(device), "cudaSetDevice");
    Check(cudaMalloc(&pointer, bytes), "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (pointer) {
      cudaSetDevice(device);
      cudaFree(pointer);
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : pointer(other.pointer), device(other.device) {
    other.pointer = nullptr;
  }
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      if (pointer) {
        cudaSetDevice(device);
        cudaFree(pointer);
      }
      pointer = other.pointer;
      device = other.device;
      other.pointer = nullptr;
    }
    return *this;
  }
  float *floats() const { return static_cast<float *>(pointer); }
};

struct DeviceWeight {
  const void *data = nullptr;
  int rows = 0, cols = 0, type = 0;
};
struct DeviceLayer {
  const float *norm = nullptr;
  const float *ffn_norm = nullptr;
  const float *bq = nullptr;
  const float *bk = nullptr;
  const float *bv = nullptr;
  DeviceWeight q, k, v, o, gate, up, down;
};

constexpr int kThreads = 256;

__device__ float WeightValue(DeviceWeight w, int row, int col) {
  if (w.type == 0)
    return static_cast<const float *>(
        w.data)[static_cast<size_t>(row) * w.cols + col];
  const int block_bytes = w.type == 8 ? 34 : w.type == 2 ? 18 : 20;
  const size_t blocks = (static_cast<size_t>(w.cols) + 31) / 32;
  const auto *block =
      static_cast<const unsigned char *>(w.data) +
      (static_cast<size_t>(row) * blocks + col / 32) * block_bytes;
  const float scale = __half2float(*reinterpret_cast<const __half *>(block));
  const int i = col % 32;
  if (w.type == 8)
    return scale * reinterpret_cast<const signed char *>(block + 2)[i];
  const int byte = block[(w.type == 3 ? 4 : 2) + i % 16];
  const int q = i < 16 ? byte & 15 : byte >> 4;
  return w.type == 3
             ? scale * q +
                   __half2float(*reinterpret_cast<const __half *>(block + 2))
             : scale * (q - 8);
}

__global__ void LinearKernel(const float *x, DeviceWeight w, const float *bias,
                             float *out) {
  const int row = blockIdx.x, tid = threadIdx.x;
  __shared__ float values[kThreads];
  float sum = 0;
  for (int col = tid; col < w.cols; col += blockDim.x)
    sum += x[col] * WeightValue(w, row, col);
  values[tid] = sum;
  __syncthreads();
  for (int width = blockDim.x / 2; width; width /= 2) {
    if (tid < width)
      values[tid] += values[tid + width];
    __syncthreads();
  }
  if (tid == 0)
    out[row] = values[0] + (bias ? bias[row] : 0);
}

__global__ void EmbedKernel(const float *embedding, int token, int dim,
                            float *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim)
    out[i] = embedding[static_cast<size_t>(token) * dim + i];
}

__global__ void NormKernel(const float *x, const float *weight, int dim,
                           float eps, float *out) {
  __shared__ float squares[kThreads];
  const int tid = threadIdx.x;
  float sum = 0;
  for (int i = tid; i < dim; i += blockDim.x)
    sum += x[i] * x[i];
  squares[tid] = sum;
  __syncthreads();
  for (int width = blockDim.x / 2; width; width /= 2) {
    if (tid < width)
      squares[tid] += squares[tid + width];
    __syncthreads();
  }
  const float inv = rsqrtf(squares[0] / dim + eps);
  for (int i = tid; i < dim; i += blockDim.x)
    out[i] = x[i] * inv * weight[i];
}

__global__ void RopeKernel(float *x, int heads, int hd, int pos, float theta,
                           bool neox) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= heads * (hd / 2))
    return;
  const int head = pair / (hd / 2), i = pair % (hd / 2);
  const int a = head * hd + (neox ? i : 2 * i);
  const int b = a + (neox ? hd / 2 : 1);
  const float angle = pos / powf(theta, 2.0f * i / hd);
  const float first = x[a], second = x[b];
  float sine, cosine;
  sincosf(angle, &sine, &cosine);
  x[a] = first * cosine - second * sine;
  x[b] = first * sine + second * cosine;
}

__global__ void StoreKvKernel(const float *k, const float *v, float *keys,
                              float *values, size_t offset, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    keys[offset + i] = k[i];
    values[offset + i] = v[i];
  }
}

__global__ void ScoresKernel(const float *q, const float *keys, float *scores,
                             int pos, int hd, int heads, int kv_heads,
                             int max_seq, int layer) {
  const int h = blockIdx.x / (pos + 1), t = blockIdx.x % (pos + 1),
            tid = threadIdx.x;
  const int kh = h / (heads / kv_heads);
  const size_t offset =
      ((static_cast<size_t>(layer) * max_seq + t) * kv_heads + kh) * hd;
  __shared__ float dots[kThreads];
  float sum = 0;
  for (int d = tid; d < hd; d += blockDim.x)
    sum += q[h * hd + d] * keys[offset + d];
  dots[tid] = sum;
  __syncthreads();
  for (int width = blockDim.x / 2; width; width /= 2) {
    if (tid < width)
      dots[tid] += dots[tid + width];
    __syncthreads();
  }
  if (tid == 0)
    scores[static_cast<size_t>(h) * max_seq + t] =
        dots[0] * rsqrtf(static_cast<float>(hd));
}

__global__ void SoftmaxKernel(float *scores, int count, int stride) {
  float *row = scores + static_cast<size_t>(blockIdx.x) * stride;
  const int tid = threadIdx.x;
  __shared__ float scratch[kThreads];
  float maximum = -FLT_MAX;
  for (int i = tid; i < count; i += blockDim.x)
    maximum = fmaxf(maximum, row[i]);
  scratch[tid] = maximum;
  __syncthreads();
  for (int width = blockDim.x / 2; width; width /= 2) {
    if (tid < width)
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + width]);
    __syncthreads();
  }
  maximum = scratch[0];
  // 所有线程读完最大值后再复用归约缓冲区。
  __syncthreads();
  float sum = 0;
  for (int i = tid; i < count; i += blockDim.x) {
    row[i] = expf(row[i] - maximum);
    sum += row[i];
  }
  scratch[tid] = sum;
  __syncthreads();
  for (int width = blockDim.x / 2; width; width /= 2) {
    if (tid < width)
      scratch[tid] += scratch[tid + width];
    __syncthreads();
  }
  for (int i = tid; i < count; i += blockDim.x)
    row[i] /= scratch[0];
}

__global__ void AttentionKernel(const float *scores, const float *values,
                                float *out, int pos, int hd, int heads,
                                int kv_heads, int max_seq, int layer) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= heads * hd)
    return;
  const int h = i / hd, d = i % hd, kh = h / (heads / kv_heads);
  float sum = 0;
  for (int t = 0; t <= pos; ++t) {
    const size_t offset =
        ((static_cast<size_t>(layer) * max_seq + t) * kv_heads + kh) * hd + d;
    sum += scores[static_cast<size_t>(h) * max_seq + t] * values[offset];
  }
  out[i] = sum;
}

__global__ void AddKernel(float *x, const float *y, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] += y[i];
}
__global__ void SwiGluKernel(float *gate, const float *up, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

int Grid(int count) { return (count + kThreads - 1) / kThreads; }
} // namespace

struct CudaModelState {
  int device;
  std::vector<DeviceBuffer> allocations;
  std::vector<DeviceLayer> layers;
  const float *embedding = nullptr;
  const float *final_norm = nullptr;
  DeviceWeight output;
  size_t uploaded_bytes = 0;
  int uploaded_weights = 0;

  explicit CudaModelState(int id) : device(id) {}
  const void *Upload(const void *data, size_t bytes) {
    allocations.emplace_back(bytes, device);
    Check(cudaMemcpy(allocations.back().pointer, data, bytes,
                     cudaMemcpyHostToDevice),
          "upload weight");
    uploaded_bytes += bytes;
    return allocations.back().pointer;
  }
  const float *UploadFloat(const Tensor &tensor) {
    if (tensor.data.empty())
      return nullptr;
    return static_cast<const float *>(
        Upload(tensor.data.data(), tensor.data.size() * sizeof(float)));
  }
  DeviceWeight UploadWeight(const Tensor &tensor) {
    ++uploaded_weights;
    if (tensor.quantized) {
      const auto &q = *tensor.quantized;
      return {Upload(q.bytes.data(), q.bytes.size()), q.rows, q.cols,
              static_cast<int>(q.type)};
    }
    return {UploadFloat(tensor), tensor.shape[0], tensor.shape[1], 0};
  }
};

struct CudaContextState {
  std::shared_ptr<CudaModelState> owner;
  DeviceBuffer keys, values, x, norm, q, k, v, attention, temporary, gate, up,
      logits, scores;
  unsigned long long linear_calls = 0, activation_calls = 0,
                     attention_calls = 0, d2h_bytes = 0;

  CudaContextState(const std::shared_ptr<CudaModelState> &model,
                   const ModelConfig &c)
      : owner(model) {
    const size_t kv_dim = static_cast<size_t>(c.n_kv_heads) * c.head_dim;
    if (kv_dim >
        std::numeric_limits<size_t>::max() / c.n_layers / c.max_seq_len)
      throw std::runtime_error("CUDA KV cache size overflow");
    const size_t elements = kv_dim * c.n_layers * c.max_seq_len;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float))
      throw std::runtime_error("CUDA KV cache size overflow");
    auto alloc = [&](size_t n) {
      return DeviceBuffer(n * sizeof(float), model->device);
    };
    keys = alloc(elements);
    values = alloc(elements);
    Check(cudaMemset(keys.pointer, 0, elements * sizeof(float)), "clear keys");
    Check(cudaMemset(values.pointer, 0, elements * sizeof(float)),
          "clear values");
    x = alloc(c.dim);
    norm = alloc(c.dim);
    q = alloc(c.dim);
    k = alloc(kv_dim);
    v = alloc(kv_dim);
    attention = alloc(c.dim);
    temporary = alloc(c.dim);
    gate = alloc(c.hidden_dim);
    up = alloc(c.hidden_dim);
    logits = alloc(c.vocab_size);
    scores = alloc(static_cast<size_t>(c.n_heads) * c.max_seq_len);
  }
};

void EnableCuda(MiniLlamaModel &model, int device) {
  if (!model.loaded)
    throw std::runtime_error("cannot upload an unloaded model");
  int count = 0;
  Check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (device < 0 || device >= count)
    throw std::runtime_error("CUDA device id out of range");
  auto gpu = std::make_shared<CudaModelState>(device);
  gpu->embedding = gpu->UploadFloat(model.token_embedding);
  gpu->final_norm = gpu->UploadFloat(model.final_norm);
  gpu->output = gpu->UploadWeight(model.lm_head);
  for (const auto &layer : model.layers) {
    DeviceLayer uploaded;
    uploaded.norm = gpu->UploadFloat(layer.attention_norm);
    uploaded.ffn_norm = gpu->UploadFloat(layer.ffn_norm);
    uploaded.bq = gpu->UploadFloat(layer.bq);
    uploaded.bk = gpu->UploadFloat(layer.bk);
    uploaded.bv = gpu->UploadFloat(layer.bv);
    uploaded.q = gpu->UploadWeight(layer.wq);
    uploaded.k = gpu->UploadWeight(layer.wk);
    uploaded.v = gpu->UploadWeight(layer.wv);
    uploaded.o = gpu->UploadWeight(layer.wo);
    uploaded.gate = gpu->UploadWeight(layer.w_gate);
    uploaded.up = gpu->UploadWeight(layer.w_up);
    uploaded.down = gpu->UploadWeight(layer.w_down);
    gpu->layers.push_back(uploaded);
  }
  model.cuda = std::move(gpu);
}

std::string CudaDeviceInfo() {
  int count = 0;
  Check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  std::ostringstream out;
  out << "CUDA devices=" << count;
  for (int device = 0; device < count; ++device) {
    cudaDeviceProp prop{};
    Check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    out << '\n'
        << device << ": " << prop.name
        << " memory_bytes=" << prop.totalGlobalMem << " compute=" << prop.major
        << '.' << prop.minor;
  }
  return out.str();
}

Tensor ForwardCuda(MiniLlamaContext &context, const MiniLlamaModel &model,
                   int token) {
  const auto &c = model.config;
  Check(cudaSetDevice(model.cuda->device), "cudaSetDevice");
  if (!context.cuda)
    context.cuda = std::make_shared<CudaContextState>(model.cuda, c);
  auto &state = *context.cuda;
  if (state.owner != model.cuda)
    throw std::runtime_error("CUDA context belongs to another model");
  const auto &gpu = *model.cuda;
  auto linear = [&](const float *input, DeviceWeight weight, float *output,
                    const float *bias = nullptr) {
    LinearKernel<<<weight.rows, kThreads>>>(input, weight, bias, output);
    ++state.linear_calls;
  };
  auto norm = [&](const float *input, const float *weight, float *output) {
    NormKernel<<<1, kThreads>>>(input, weight, c.dim, c.rms_norm_eps, output);
    ++state.activation_calls;
  };
  EmbedKernel<<<Grid(c.dim), kThreads>>>(gpu.embedding, token, c.dim,
                                         state.x.floats());
  const int kv_dim = c.n_kv_heads * c.head_dim;
  for (int i = 0; i < c.n_layers; ++i) {
    const auto &layer = gpu.layers[i];
    norm(state.x.floats(), layer.norm, state.norm.floats());
    linear(state.norm.floats(), layer.q, state.q.floats(), layer.bq);
    linear(state.norm.floats(), layer.k, state.k.floats(), layer.bk);
    linear(state.norm.floats(), layer.v, state.v.floats(), layer.bv);
    const bool neox = c.rope_type == RopeType::kNeoX;
    RopeKernel<<<Grid(c.dim / 2), kThreads>>>(state.q.floats(), c.n_heads,
                                              c.head_dim, context.pos,
                                              c.rope_theta, neox);
    RopeKernel<<<Grid(kv_dim / 2), kThreads>>>(state.k.floats(), c.n_kv_heads,
                                               c.head_dim, context.pos,
                                               c.rope_theta, neox);
    const size_t offset =
        (static_cast<size_t>(i) * c.max_seq_len + context.pos) * kv_dim;
    StoreKvKernel<<<Grid(kv_dim), kThreads>>>(
        state.k.floats(), state.v.floats(), state.keys.floats(),
        state.values.floats(), offset, kv_dim);
    ScoresKernel<<<static_cast<unsigned>(c.n_heads) * (context.pos + 1),
                   kThreads>>>(state.q.floats(), state.keys.floats(),
                               state.scores.floats(), context.pos, c.head_dim,
                               c.n_heads, c.n_kv_heads, c.max_seq_len, i);
    SoftmaxKernel<<<c.n_heads, kThreads>>>(state.scores.floats(),
                                           context.pos + 1, c.max_seq_len);
    AttentionKernel<<<Grid(c.dim), kThreads>>>(
        state.scores.floats(), state.values.floats(), state.attention.floats(),
        context.pos, c.head_dim, c.n_heads, c.n_kv_heads, c.max_seq_len, i);
    ++state.attention_calls;
    linear(state.attention.floats(), layer.o, state.temporary.floats());
    AddKernel<<<Grid(c.dim), kThreads>>>(state.x.floats(),
                                         state.temporary.floats(), c.dim);
    norm(state.x.floats(), layer.ffn_norm, state.norm.floats());
    linear(state.norm.floats(), layer.gate, state.gate.floats());
    linear(state.norm.floats(), layer.up, state.up.floats());
    SwiGluKernel<<<Grid(c.hidden_dim), kThreads>>>(
        state.gate.floats(), state.up.floats(), c.hidden_dim);
    linear(state.gate.floats(), layer.down, state.temporary.floats());
    AddKernel<<<Grid(c.dim), kThreads>>>(state.x.floats(),
                                         state.temporary.floats(), c.dim);
    state.activation_calls += 5;
  }
  norm(state.x.floats(), gpu.final_norm, state.norm.floats());
  linear(state.norm.floats(), gpu.output, state.logits.floats());
  Check(cudaGetLastError(), "CUDA forward kernel launch");
  Tensor output({c.vocab_size});
  // 采样仍在 CPU 上执行，每步只把最终 logits 拷回主机。
  Check(cudaMemcpy(output.data.data(), state.logits.pointer,
                   output.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download logits");
  state.d2h_bytes += output.size() * sizeof(float);
  return output;
}

std::string CudaStats(const MiniLlamaModel &model,
                      const MiniLlamaContext &context) {
  if (!model.cuda)
    return "";
  std::ostringstream out;
  out << "uploaded_weights=" << model.cuda->uploaded_weights
      << " uploaded_bytes=" << model.cuda->uploaded_bytes;
  if (context.cuda)
    out << " cuda_linear_calls=" << context.cuda->linear_calls
        << " cuda_activation_calls=" << context.cuda->activation_calls
        << " cuda_attention_calls=" << context.cuda->attention_calls
        << " device_to_host_bytes=" << context.cuda->d2h_bytes
        << " cpu_attention_fallback_calls=0";
  return out.str();
}
} // namespace mini_llama
