#include "mini_llama/runtime.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/loader.h"
#include "mini_llama/prompt_builder.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace mini_llama {
InferenceModel LoadInferenceModel(const std::string &source,
                                  const std::string &config_path,
                                  const std::string &tokenizer_path) {
  namespace fs = std::filesystem;
  fs::path path(source), config(config_path);
  if (fs::is_directory(path)) {
    if (fs::exists(path / "model.json") && fs::exists(path / "model.bin")) {
      config = path / "model.json";
      path /= "model.bin";
    } else {
      std::vector<fs::path> candidates;
      for (const auto &entry : fs::directory_iterator(path))
        if (entry.is_regular_file() && entry.path().extension() == ".gguf")
          candidates.push_back(entry.path());
      if (candidates.size() != 1)
        throw std::runtime_error("model directory must contain exactly one "
                                 "GGUF or model.json/model.bin");
      path = candidates.front();
    }
  }
  InferenceModel loaded;
  loaded.source = path.string();
  if (path.extension() == ".gguf") {
    loaded.model = LoadGgufModel(path.string());
    if (!loaded.model.loaded)
      throw std::runtime_error(loaded.model.load_error);
    loaded.tokenizer = CreateGgufTokenizer(path.string());
    loaded.chat_template = LoadChatTemplateFromGguf(path.string());
  } else {
    const auto manifest = ParseManifest(config.string());
    loaded.model = LoadModel(config.string(), path.string());
    if (!loaded.model.loaded)
      throw std::runtime_error(loaded.model.load_error);
    if (manifest.tokenizer.type == "json_vocab") {
      fs::path vocab(manifest.tokenizer.path);
      if (vocab.is_relative())
        vocab = config.parent_path() / vocab;
      loaded.tokenizer = std::make_unique<JsonVocabTokenizer>(vocab.string());
    } else
      loaded.tokenizer = std::make_unique<AsciiTokenizer>();
    if (loaded.tokenizer->bos_id() != manifest.tokenizer.bos_id ||
        loaded.tokenizer->eos_id() != manifest.tokenizer.eos_id ||
        loaded.tokenizer->unk_id() != manifest.tokenizer.unk_id)
      throw std::runtime_error("tokenizer special ids do not match manifest");
  }
  // 显式指定的词表加载失败时直接报错，避免换成另一种分词方式。
  if (!tokenizer_path.empty())
    loaded.tokenizer = std::make_unique<JsonVocabTokenizer>(tokenizer_path);
  if (!loaded.tokenizer)
    throw std::runtime_error("failed to load model tokenizer");
  if (loaded.tokenizer->vocab_size() > loaded.model.config.vocab_size)
    throw std::runtime_error("tokenizer vocabulary exceeds model vocabulary");
  return loaded;
}
} // namespace mini_llama
