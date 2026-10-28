#include "mini_llama/backend.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "tests/test_main.h"

using namespace mini_llama;

static bool TestCudaForward() {
#ifdef MINI_AI_CUDA
  // CUDA 构建需要真实设备，初始化失败也应让测试失败。
  std::cout << CudaDeviceInfo() << '\n';
  for (RopeType layout : {RopeType::kNormal, RopeType::kNeoX}) {
    for (int type : {0, 8, 2, 3}) {
      auto cpu = LoadModel("models/tiny/model.json", "models/tiny/model.bin");
      cpu.config.rope_type = layout;
      for (auto &layer : cpu.layers) {
        layer.bq = Tensor({cpu.config.dim}, 0.1f);
        const int kv = cpu.config.n_kv_heads * cpu.config.head_dim;
        layer.bk = Tensor({kv}, -0.2f);
        layer.bv = Tensor({kv}, 0.3f);
      }
      if (type)
        QuantizeModel(cpu, static_cast<QuantType>(type));
      auto gpu = cpu;
      EnableCuda(gpu);
      MiniLlamaContext a(&cpu), b(&gpu);
      for (int pos = 0; pos < 5; ++pos) {
        a.pos = b.pos = pos;
        const auto expected = ForwardToken(a, cpu, 65 + pos);
        const auto actual = ForwardToken(b, gpu, 65 + pos);
        for (size_t i = 0; i < expected.size(); ++i)
          MINI_LLAMA_ASSERT_NEAR(actual[i], expected[i], 1e-4f);
      }
      std::cout << CudaStats(gpu, b) << '\n';
    }
  }
#else
  bool rejected = false;
  MiniLlamaModel model;
  try {
    EnableCuda(model);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  MINI_LLAMA_ASSERT_TRUE(rejected);
#endif
  return true;
}

static struct CudaRegistrar {
  CudaRegistrar() {
    RegisterTest("cuda_backend_availability_and_forward", TestCudaForward);
  }
} cuda_registrar;
