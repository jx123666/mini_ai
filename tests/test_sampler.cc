//测采样：greedy、温度、top-k、固定 seed、非法参数
#include <limits>
#include <stdexcept>

#include "mini_llama/sampler.h"
#include "mini_llama/tensor.h"
#include "tests/test_main.h"

using mini_llama::MiniSampler;
using mini_llama::SampleGreedy;
using mini_llama::SamplingParams;
using mini_llama::Tensor;

//greedy：并列最大值取第一个
static bool TestSamplerGreedy() {
  Tensor logits({5}, 0.0f);
  logits[0] = 1.0f;
  logits[1] = 5.0f;
  logits[2] = 3.0f;
  logits[3] = 5.0f;
  logits[4] = 2.0f;

  int idx = MiniSampler::SampleGreedy(logits);
  MINI_LLAMA_ASSERT_EQ(idx, 1);
  return true;
}

//temperature=0 应等价于 greedy
static bool TestSamplerTemperatureZeroIsGreedy() {
  Tensor logits({5}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 2.0f;
  logits[2] = 1.0f;
  logits[3] = 0.5f;
  logits[4] = 0.2f;

  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = 0.0f;
  params.top_k = 0;

  int expected = MiniSampler::SampleGreedy(logits);
  for (int i = 0; i < 20; ++i) {
    int idx = sampler.Sample(logits, params);
    MINI_LLAMA_ASSERT_EQ(idx, expected);
  }
  return true;
}

//top_k=1 也应退回 greedy
static bool TestSamplerTopKOneIsGreedy() {
  Tensor logits({5}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 2.0f;
  logits[2] = 1.0f;
  logits[3] = 0.5f;
  logits[4] = 0.2f;

  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = 1;

  int expected = MiniSampler::SampleGreedy(logits);
  for (int i = 0; i < 20; ++i) {
    int idx = sampler.Sample(logits, params);
    MINI_LLAMA_ASSERT_EQ(idx, expected);
  }
  return true;
}

//同样 seed，温度采样序列应一致
static bool TestSamplerTemperatureReproducible() {
  Tensor logits({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    logits[i] = static_cast<float>(i) * 0.5f;
  }

  MiniSampler sampler1(123);
  MiniSampler sampler2(123);
  SamplingParams params;
  params.temperature = 0.8f;
  params.top_k = 0;

  for (int i = 0; i < 10; ++i) {
    int a = sampler1.Sample(logits, params);
    int b = sampler2.Sample(logits, params);
    MINI_LLAMA_ASSERT_EQ(a, b);
  }
  return true;
}

//同样 seed，top-k 序列也应一致
static bool TestSamplerTopKReproducible() {
  Tensor logits({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    logits[i] = static_cast<float>(i) * 0.5f;
  }

  MiniSampler sampler1(456);
  MiniSampler sampler2(456);
  SamplingParams params;
  params.temperature = 0.8f;
  params.top_k = 3;

  for (int i = 0; i < 10; ++i) {
    int a = sampler1.Sample(logits, params);
    int b = sampler2.Sample(logits, params);
    MINI_LLAMA_ASSERT_EQ(a, b);
  }
  return true;
}

//不同 seed 大概率抽到不同序列
static bool TestSamplerDifferentSeedsDiffer() {
  Tensor logits({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    logits[i] = static_cast<float>(i) * 0.3f;
  }

  MiniSampler sampler1(1);
  MiniSampler sampler2(2);
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = 0;

  int same_count = 0;
  for (int i = 0; i < 20; ++i) {
    int a = sampler1.Sample(logits, params);
    int b = sampler2.Sample(logits, params);
    if (a == b) {
      ++same_count;
    }
  }
  MINI_LLAMA_ASSERT_TRUE(same_count < 18);
  return true;
}

//top_k=2 时只能抽到分数最高的两个
static bool TestSamplerTopKLimitsCandidates() {
  Tensor logits({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    logits[i] = static_cast<float>(i);
  }

  MiniSampler sampler(789);
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = 2;

  for (int i = 0; i < 50; ++i) {
    int idx = sampler.Sample(logits, params);
    MINI_LLAMA_ASSERT_TRUE(idx == 8 || idx == 9);
  }
  return true;
}

static bool TestSamplerLegacyGreedy() {
  Tensor logits({5}, 0.0f);
  logits[0] = 1.0f;
  logits[1] = 5.0f;
  logits[2] = 3.0f;
  logits[3] = 4.0f;
  logits[4] = 2.0f;

  int idx = SampleGreedy(logits);
  MINI_LLAMA_ASSERT_EQ(idx, 1);
  return true;
}

//默认参数是 greedy
static bool TestSamplerDefaultParamsAreGreedy() {
  Tensor logits({5}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 2.0f;
  logits[2] = 1.0f;
  logits[3] = 0.5f;
  logits[4] = 0.2f;

  MiniSampler sampler(42);
  SamplingParams params;

  int idx = sampler.Sample(logits, params);
  MINI_LLAMA_ASSERT_EQ(idx, MiniSampler::SampleGreedy(logits));
  return true;
}

static bool TestSamplerRejectsNegativeTopK() {
  Tensor logits({3}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 0.2f;
  logits[2] = 0.3f;

  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = -1;

  try {
    sampler.Sample(logits, params);
    MINI_LLAMA_ASSERT_FAIL("expected exception for negative top_k");
  } catch (const std::runtime_error&) {
    return true;
  }
}

static bool TestSamplerRejectsNanTemperature() {
  Tensor logits({3}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 0.2f;
  logits[2] = 0.3f;

  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = std::numeric_limits<float>::quiet_NaN();
  params.top_k = 0;

  try {
    sampler.Sample(logits, params);
    MINI_LLAMA_ASSERT_FAIL("expected exception for NaN temperature");
  } catch (const std::runtime_error&) {
    return true;
  }
}

static bool TestSamplerRejectsEmptyLogits() {
  Tensor logits;
  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = 0.0f;

  try {
    sampler.Sample(logits, params);
    MINI_LLAMA_ASSERT_FAIL("expected exception for empty logits");
  } catch (const std::runtime_error&) {
    return true;
  }
}

static bool TestSamplerRejectsNonfiniteLogits() {
  Tensor logits({3}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = std::numeric_limits<float>::infinity();
  logits[2] = 0.3f;

  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = 1.0f;

  try {
    sampler.Sample(logits, params);
    MINI_LLAMA_ASSERT_FAIL("expected exception for non-finite logits");
  } catch (const std::runtime_error&) {
    return true;
  }
}

static bool TestSamplerTopKLargerThanVocabIsClamped() {
  Tensor logits({4}, 0.0f);
  logits[0] = 0.0f;
  logits[1] = 1.0f;
  logits[2] = 2.0f;
  logits[3] = 3.0f;

  MiniSampler sampler(42);
  for (int i = 0; i < 20; ++i) {
    int idx = sampler.SampleTopK(logits, 1.0f, 100);
    MINI_LLAMA_ASSERT_TRUE(idx >= 0 && idx < 4);
  }
  return true;
}

static bool TestSamplerDirectTopKZeroRejected() {
  Tensor logits({3}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 0.2f;
  logits[2] = 0.3f;

  MiniSampler sampler(42);
  try {
    sampler.SampleTopK(logits, 1.0f, 0);
    MINI_LLAMA_ASSERT_FAIL("expected exception for direct top_k zero");
  } catch (const std::runtime_error&) {
    return true;
  }
}

//并列最高分时，top-k 按 id 小的先取
static bool TestSamplerTopKTiesKeepExactK() {
  Tensor logits({5}, 0.0f);
  logits[0] = 5.0f;
  logits[1] = 5.0f;
  logits[2] = 5.0f;
  logits[3] = 1.0f;
  logits[4] = 0.0f;

  MiniSampler sampler(321);
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = 2;

  for (int i = 0; i < 40; ++i) {
    int idx = sampler.Sample(logits, params);
    MINI_LLAMA_ASSERT_TRUE(idx == 0 || idx == 1);
  }
  return true;
}

static struct SamplerTestRegistrar {
  SamplerTestRegistrar() {
    RegisterTest("sampler_greedy", TestSamplerGreedy);
    RegisterTest("sampler_temperature_zero_is_greedy",
                 TestSamplerTemperatureZeroIsGreedy);
    RegisterTest("sampler_top_k_one_is_greedy", TestSamplerTopKOneIsGreedy);
    RegisterTest("sampler_temperature_reproducible",
                 TestSamplerTemperatureReproducible);
    RegisterTest("sampler_top_k_reproducible", TestSamplerTopKReproducible);
    RegisterTest("sampler_different_seeds_differ",
                 TestSamplerDifferentSeedsDiffer);
    RegisterTest("sampler_top_k_limits_candidates",
                 TestSamplerTopKLimitsCandidates);
    RegisterTest("sampler_legacy_greedy", TestSamplerLegacyGreedy);
    RegisterTest("sampler_default_params_are_greedy",
                 TestSamplerDefaultParamsAreGreedy);
    RegisterTest("sampler_rejects_negative_top_k",
                 TestSamplerRejectsNegativeTopK);
    RegisterTest("sampler_rejects_nan_temperature",
                 TestSamplerRejectsNanTemperature);
    RegisterTest("sampler_rejects_empty_logits", TestSamplerRejectsEmptyLogits);
    RegisterTest("sampler_rejects_nonfinite_logits",
                 TestSamplerRejectsNonfiniteLogits);
    RegisterTest("sampler_top_k_larger_than_vocab_is_clamped",
                 TestSamplerTopKLargerThanVocabIsClamped);
    RegisterTest("sampler_direct_top_k_zero_rejected",
                 TestSamplerDirectTopKZeroRejected);
    RegisterTest("sampler_top_k_ties_keep_exact_k",
                 TestSamplerTopKTiesKeepExactK);
  }
} sampler_test_registrar;
