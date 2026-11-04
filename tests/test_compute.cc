#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/thread_pool.h"
#include "src/pretokenizer.h"
#include "tests/test_main.h"

#include <atomic>
#include <cmath>
#include <stdexcept>

using namespace mini_llama;

static bool TestParallelLinear() {
  Tensor x({37}), weight({41, 37});
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = std::sin(float(i));
  for (size_t i = 0; i < weight.size(); ++i)
    weight[i] = std::cos(float(i)) * 0.1f;
  const auto reference = LinearDispatch(x, weight, MatmulMode::kNaive);
  SetMatmulThreads(3);
  for (auto mode :
       {MatmulMode::kThreaded, MatmulMode::kSimd, MatmulMode::kThreadedSimd}) {
    const auto actual = LinearDispatch(x, weight, mode);
    for (size_t i = 0; i < actual.size(); ++i)
      MINI_LLAMA_ASSERT_NEAR(actual[i], reference[i], 1e-5f);
  }
  return true;
}

static bool TestQuantizedLinear() {
  Tensor x({37}), weight({41, 37});
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = std::sin(float(i));
  for (size_t i = 0; i < weight.size(); ++i)
    weight[i] = std::cos(float(i)) * 0.1f;
  const auto original = LinearDispatch(x, weight, MatmulMode::kNaive);
  for (auto type : {QuantType::kQ80, QuantType::kQ40, QuantType::kQ41}) {
    const auto packed = QuantizeTensor(weight, type);
    MINI_LLAMA_ASSERT_TRUE(packed.data.empty());
    MINI_LLAMA_ASSERT_TRUE(packed.quantized->bytes.size() <
                           weight.size() * sizeof(float));
    const auto decoded = DequantizeTensor(packed);
    const auto reference = LinearDispatch(x, decoded, MatmulMode::kNaive);
    const auto actual = LinearDispatch(x, packed, MatmulMode::kThreadedSimd);
    for (size_t i = 0; i < actual.size(); ++i) {
      MINI_LLAMA_ASSERT_NEAR(actual[i], reference[i], 1e-5f);
      MINI_LLAMA_ASSERT_NEAR(actual[i], original[i],
                             type == QuantType::kQ80 ? 0.01f : 0.08f);
    }
  }
  return true;
}

static bool TestQuantizedRawBlocks() {
  // GGML 的 Q4_0 低半字节对应前 16 个值，高半字节对应后 16 个值。
  std::vector<uint8_t> bytes(18, 0);
  bytes[0] = 0;
  bytes[1] = 0x3c;
  for (int i = 0; i < 16; ++i)
    bytes[2 + i] = 0xf0;
  const auto decoded =
      DequantizeTensor(TensorFromQuantized(bytes, 1, 32, QuantType::kQ40));
  MINI_LLAMA_ASSERT_NEAR(decoded[0], -8.0f, 0);
  MINI_LLAMA_ASSERT_NEAR(decoded[16], 7.0f, 0);
  MINI_LLAMA_ASSERT_NEAR(HalfToFloat(1), std::ldexp(1.0f, -24), 0);
  MINI_LLAMA_ASSERT_EQ(FloatToHalf(std::ldexp(1.0f, -24)), 1);
  return true;
}

static bool TestPoolException() {
  ThreadPool pool(3);
  bool threw = false;
  try {
    pool.ParallelFor(30, [](int row) {
      if (row == 12)
        throw std::runtime_error("row failure");
    });
  } catch (const std::runtime_error &) {
    threw = true;
  }
  MINI_LLAMA_ASSERT_TRUE(threw);
  std::atomic<int> rows{0};
  pool.ParallelFor(30, [&](int) { ++rows; });
  MINI_LLAMA_ASSERT_EQ(rows.load(), 30);
  return true;
}

static bool TestPreTokenizer() {
  MINI_LLAMA_ASSERT_EQ(
      SplitPreTokens("Hello 12345!\n", "qwen2"),
      std::vector<std::string>({"Hello", " ", "123", "45", "!\n"}));
  MINI_LLAMA_ASSERT_EQ(SplitPreTokens("I'm 你好！", "qwen2"),
                       std::vector<std::string>({"I", "'m", " 你好", "！"}));
  MINI_LLAMA_ASSERT_EQ(SplitPreTokens("a  b", "gpt-2"),
                       std::vector<std::string>({"a", " ", " b"}));
  MINI_LLAMA_ASSERT_EQ(SplitPreTokens("!Hello", "qwen2"),
                       std::vector<std::string>({"!Hello"}));
  return true;
}

static struct ComputeRegistrar {
  ComputeRegistrar() {
    RegisterTest("parallel_linear_matches_scalar", TestParallelLinear);
    RegisterTest("quantized_linear_with_tail", TestQuantizedLinear);
    RegisterTest("ggml_raw_quantized_blocks", TestQuantizedRawBlocks);
    RegisterTest("thread_pool_propagates_exceptions", TestPoolException);
    RegisterTest("qwen_and_gpt_pre_tokenizer", TestPreTokenizer);
  }
} compute_registrar;
