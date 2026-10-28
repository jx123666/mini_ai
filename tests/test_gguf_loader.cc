//测GGUF映射加载：完整tiny fixture、进Forward、错误路径、可选真Qwen2
#include <filesystem>
#include <string>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf_loader.h"
#include "tests/gguf_loader_fixture.h"
#include "tests/test_main.h"

using mini_llama::ForwardToken;
using mini_llama::LayerWeights;
using mini_llama::LoadGgufModel;
using mini_llama::MiniLlamaContext;
using mini_llama::MiniLlamaModel;
using mini_llama::Tensor;

//生成tiny GGUF后检查config和各张量shape
static bool TestGgufLoaderLoadsTinyFixture() {
  std::filesystem::path path =
      mini_llama::test::WriteTinyLlamaGgufFixture("tiny_loader_fixture.gguf");
  MiniLlamaModel model = LoadGgufModel(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(model.loaded);
  MINI_LLAMA_ASSERT_TRUE(model.load_error.empty());

  MINI_LLAMA_ASSERT_EQ(model.config.vocab_size, 128);
  MINI_LLAMA_ASSERT_EQ(model.config.dim, 32);
  MINI_LLAMA_ASSERT_EQ(model.config.n_layers, 2);
  MINI_LLAMA_ASSERT_EQ(model.config.n_heads, 4);
  MINI_LLAMA_ASSERT_EQ(model.config.n_kv_heads, 4);
  MINI_LLAMA_ASSERT_EQ(model.config.head_dim, 8);
  MINI_LLAMA_ASSERT_EQ(model.config.hidden_dim, 86);
  MINI_LLAMA_ASSERT_EQ(model.config.max_seq_len, 128);
  MINI_LLAMA_ASSERT_NEAR(model.config.rms_norm_eps, 1e-5f, 1e-7f);

  //embedding应是[vocab, dim]，二维已经对调回来
  MINI_LLAMA_ASSERT_TRUE(model.token_embedding.data.size() > 0);
  MINI_LLAMA_ASSERT_EQ(model.token_embedding.shape.size(), 2);
  MINI_LLAMA_ASSERT_EQ(model.token_embedding.shape[0], 128);
  MINI_LLAMA_ASSERT_EQ(model.token_embedding.shape[1], 32);

  //每层：norm是一维，wq等是[out, in]，w_down是[dim, hidden]
  MINI_LLAMA_ASSERT_EQ(static_cast<int>(model.layers.size()), 2);
  for (int i = 0; i < 2; ++i) {
    const LayerWeights& lw = model.layers[static_cast<size_t>(i)];
    MINI_LLAMA_ASSERT_TRUE(lw.attention_norm.data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.attention_norm.shape[0], 32);

    MINI_LLAMA_ASSERT_TRUE(lw.wq.data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.wq.shape, std::vector<int>({32, 32}));
    MINI_LLAMA_ASSERT_TRUE(lw.wk.data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.wk.shape, std::vector<int>({32, 32}));
    MINI_LLAMA_ASSERT_TRUE(lw.wv.data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.wv.shape, std::vector<int>({32, 32}));
    MINI_LLAMA_ASSERT_TRUE(lw.wo.data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.wo.shape, std::vector<int>({32, 32}));

    MINI_LLAMA_ASSERT_TRUE(lw.ffn_norm.data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.ffn_norm.shape[0], 32);

    MINI_LLAMA_ASSERT_TRUE(lw.w_gate.data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.w_gate.shape, std::vector<int>({86, 32}));
    MINI_LLAMA_ASSERT_TRUE(lw.w_up.data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.w_up.shape, std::vector<int>({86, 32}));
    MINI_LLAMA_ASSERT_TRUE(lw.w_down.data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.w_down.shape, std::vector<int>({32, 86}));
  }

  MINI_LLAMA_ASSERT_TRUE(model.final_norm.data.size() > 0);
  MINI_LLAMA_ASSERT_EQ(model.final_norm.shape[0], 32);

  MINI_LLAMA_ASSERT_TRUE(model.lm_head.data.size() > 0);
  MINI_LLAMA_ASSERT_EQ(model.lm_head.shape.size(), 2);
  MINI_LLAMA_ASSERT_EQ(model.lm_head.shape[0], 128);
  MINI_LLAMA_ASSERT_EQ(model.lm_head.shape[1], 32);

  return true;
}

//映射装出来的模型要能进正常Forward
static bool TestGgufLoaderFixtureEntersForwardPath() {
  std::filesystem::path path =
      mini_llama::test::WriteTinyLlamaGgufFixture("tiny_loader_forward.gguf");
  MiniLlamaModel model = LoadGgufModel(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(model.loaded);
  //能进Forward说明权重shape和config对得上，logits长度等于词表
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  Tensor logits = ForwardToken(ctx, model, 1);
  MINI_LLAMA_ASSERT_EQ(logits.shape, std::vector<int>({128}));
  return true;
}

//检查核心config映射
static bool TestGgufLoaderLoadsRealModelWhenAvailable() {
  const std::string path = "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
  //没有这份大文件就跳过，不让干净仓库失败
  if (!std::filesystem::exists(path)) {
    return true;
  }

  MiniLlamaModel model = LoadGgufModel(path);
  MINI_LLAMA_ASSERT_TRUE(model.loaded);
  MINI_LLAMA_ASSERT_TRUE(model.load_error.empty());
  MINI_LLAMA_ASSERT_EQ(model.config.vocab_size, 151936);
  MINI_LLAMA_ASSERT_EQ(model.config.dim, 896);
  MINI_LLAMA_ASSERT_EQ(model.config.n_layers, 24);
  MINI_LLAMA_ASSERT_EQ(model.config.n_heads, 14);
  MINI_LLAMA_ASSERT_EQ(model.config.n_kv_heads, 2);
  MINI_LLAMA_ASSERT_EQ(model.config.head_dim, 64);
  MINI_LLAMA_ASSERT_EQ(model.config.hidden_dim, 4864);
  MINI_LLAMA_ASSERT_EQ(model.config.max_seq_len, 32768);
  MINI_LLAMA_ASSERT_NEAR(model.config.rms_norm_eps, 1e-6f, 1e-7f);
  return true;
}

//文件不存在应失败，并把原因写进load_error
static bool TestGgufLoaderMissingFile() {
  MiniLlamaModel model = LoadGgufModel("models/chat/nonexistent.gguf");
  MINI_LLAMA_ASSERT_TRUE(!model.loaded);
  MINI_LLAMA_ASSERT_TRUE(!model.load_error.empty());
  return true;
}

//json不是GGUF，magic对不上就失败
static bool TestGgufLoaderInvalidFile() {
  MiniLlamaModel model = LoadGgufModel("models/tiny/model.json");
  MINI_LLAMA_ASSERT_TRUE(!model.loaded);
  MINI_LLAMA_ASSERT_TRUE(!model.load_error.empty());
  return true;
}

//test.gguf只有结构没有完整config，应在BuildConfig阶段失败
static bool TestGgufLoaderRejectsIncompleteConfig() {
  MiniLlamaModel model = LoadGgufModel("models/tiny/test.gguf");
  MINI_LLAMA_ASSERT_TRUE(!model.loaded);
  MINI_LLAMA_ASSERT_TRUE(!model.load_error.empty());
  return true;
}

//启动时把这组测试登记进极简框架
static bool TestQwenBiasRoundtrip() {
  auto source = mini_llama::LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  source.config.rope_type = mini_llama::RopeType::kNeoX;
  for (auto& layer : source.layers) {
    layer.bq = mini_llama::Tensor({source.config.dim}, 0.125f);
    const int kv_dim = source.config.n_kv_heads * source.config.head_dim;
    layer.bk = mini_llama::Tensor({kv_dim}, -0.25f);
    layer.bv = mini_llama::Tensor({kv_dim}, 0.5f);
  }
  const auto path = mini_llama::test::WriteModelGgufFixture("qwen_bias.gguf", source, "qwen2");
  const auto loaded = LoadGgufModel(path.string());
  std::filesystem::remove(path);
  MINI_LLAMA_ASSERT_TRUE(loaded.loaded);
  MINI_LLAMA_ASSERT_TRUE(loaded.config.rope_type == mini_llama::RopeType::kNeoX);
  MINI_LLAMA_ASSERT_NEAR(loaded.layers[0].bq[0], 0.125f, 1e-7f);
  MiniLlamaContext a(&source), b(&loaded);
  for (int pos = 0; pos < 3; ++pos) {
    a.pos = b.pos = pos;
    const auto expected = mini_llama::ForwardToken(a, source, 65 + pos);
    const auto actual = mini_llama::ForwardToken(b, loaded, 65 + pos);
    for (size_t i = 0; i < expected.size(); ++i) {
      MINI_LLAMA_ASSERT_NEAR(actual[i], expected[i], 1e-6f);
    }
  }
  return true;
}

static struct GgufLoaderTestRegistrar {
  GgufLoaderTestRegistrar() {
    RegisterTest("qwen_bias_and_rope_roundtrip", TestQwenBiasRoundtrip);
    RegisterTest("gguf_loader_loads_tiny_fixture",
                 TestGgufLoaderLoadsTinyFixture);
    RegisterTest("gguf_loader_fixture_enters_forward_path",
                 TestGgufLoaderFixtureEntersForwardPath);
    RegisterTest("gguf_loader_loads_real_model_when_available",
                 TestGgufLoaderLoadsRealModelWhenAvailable);
    RegisterTest("gguf_loader_missing_file", TestGgufLoaderMissingFile);
    RegisterTest("gguf_loader_invalid_file", TestGgufLoaderInvalidFile);
    RegisterTest("gguf_loader_rejects_incomplete_config",
                 TestGgufLoaderRejectsIncompleteConfig);
  }
} gguf_loader_test_registrar;
