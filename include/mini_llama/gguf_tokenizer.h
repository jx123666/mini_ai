//GGUF分词器：从文件metadata读tokens / merges / special id
//合并算法和BpeTokenizer同一套，只是数据来源换成真实GGUF
#ifndef INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
#define INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_

#include <cstdint>
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
  //从.gguf读tokenizer.ggml.*；缺词表返回false
  bool Load(const std::string& gguf_path);

  //测试入口：词表和merges已经拆出来了
  //merges每一项形如"h e"，越靠前越先合并
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
  //特殊串按长度从长到短，编码时优先整段吃掉
  std::vector<std::pair<std::string, int>> special_tokens_;
  std::vector<std::string> b2u_;
  std::unordered_map<std::string, uint8_t> u2b_;
  int bos_id_ = -1;
  int eos_id_ = -1;
  int unk_id_ = -1;
  std::string pre_tokenizer_;
  bool add_bos_ = false;
  bool add_eos_ = false;

  void ResetByteMap();
  void IndexTokens(const std::vector<std::string>& tokens,
                   const std::vector<int32_t>* token_types);
  void IndexMerges(const std::vector<std::string>& merges);
};

//打开gguf并读分词器；失败返回空指针
std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path);

}

#endif
