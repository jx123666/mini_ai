// =============================================================================
// 文件作用：gguf_tokenizer.h
//   GGUF 模型文件里已经打包了分词器。真实工程从这些字段读取：
//     tokenizer.ggml.tokens       每个 id 对应的字符串片段
//     tokenizer.ggml.merges       BPE 合并规则，如 "h e"
//     tokenizer.ggml.token_type   哪些是特殊符号
//     tokenizer.ggml.bos_token_id / eos_token_id
//
// 读出来之后，合并算法和 BpeTokenizer 完全一样。
// 本学习版不解析 GGUF 二进制（那是后面「模型文件」的任务），
// 用 LoadPieces() 直接喂入同一套词表和 merges，方便把合并逻辑测清楚。
// =============================================================================

#ifndef INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
#define INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mini_llama/tokenizer.h"

namespace mini_llama {

class GgufTokenizer : public ITokenizer {
 public:
  // 教学入口：词表、合并规则、特殊 id 已经从文件里取出来了。
  // merges 每一项形如 "h e"，顺序就是优先级（越靠前越先合并）。
  bool LoadPieces(const std::vector<std::string>& tokens,
                  const std::vector<std::string>& merges, int bos_id,
                  int eos_id, int unk_id);

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override {
    return static_cast<int>(id_to_token_.size());
  }
  int bos_id() const override { return bos_id_; }
  int eos_id() const override { return eos_id_; }
  int unk_id() const override { return unk_id_; }

 private:
  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> id_to_token_;
  std::map<std::pair<std::string, std::string>, int> merge_ranks_;
  std::vector<std::pair<std::string, int>> special_tokens_;
  std::vector<std::string> b2u_;
  std::unordered_map<std::string, uint8_t> u2b_;
  int bos_id_ = -1;
  int eos_id_ = -1;
  int unk_id_ = -1;
};

// 完整 mini-llama 会打开 .gguf 并读 metadata。
// 这里还没有 GGUF 解析器，所以这个工厂目前返回空指针。
std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
