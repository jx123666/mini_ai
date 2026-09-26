// =============================================================================
// 文件作用：tokenizer.h
//   定义「文本 <-> token id」的接口，以及本任务要掌握的三类分词器。
//
// 模型不认识字符串，只认识整数 id。Tokenizer 做两件事：
//   Encode：把 "hello" 变成 [1, 104, 101, ...] 这种 id 序列
//   Decode：把 id 序列变回能打印的文字
//
// 本文件里的三个类，由浅到深：
//   1) AsciiTokenizer      一个字符一个 id（验收：能实现）
//   2) JsonVocabTokenizer  从词表文件按「最长匹配」切词
//   3) BpeTokenizer        按合并规则把相邻片段拼成更长的词（验收：理解合并）
// =============================================================================

#ifndef INCLUDE_MINI_LLAMA_TOKENIZER_H_
#define INCLUDE_MINI_LLAMA_TOKENIZER_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini_llama {

// 所有分词器都要能编码、解码、报告词表大小和几个特殊 id。
class ITokenizer {
 public:
  virtual ~ITokenizer() = default;

  // 文本 -> token id。ASCII / JSON 词表版会在最前面加 BOS。
  virtual std::vector<int> Encode(const std::string& text) const = 0;

  // 单个 id -> 字符串片段
  virtual std::string DecodeToken(int token) const = 0;

  // 一串 id -> 拼起来的字符串
  virtual std::string Decode(const std::vector<int>& tokens) const = 0;

  virtual int vocab_size() const = 0;
  virtual int bos_id() const = 0;  // Begin Of Sequence，序列开始
  virtual int eos_id() const = 0;  // End Of Sequence，序列结束
  virtual int unk_id() const = 0;  // Unknown，词表里没有的字符
};

// ---------------------------------------------------------------------------
// ASCII 分词器：词表固定 128 个，id 就是字符本身的编号。
//   'h' 的 ASCII 码是 104，所以它的 token id 就是 104。
//   0 = <unk>，1 = <bos>，2 = <eos>（和字符码有重叠，见 tokenizer.cc 注释）
// ---------------------------------------------------------------------------
class AsciiTokenizer : public ITokenizer {
 public:
  static constexpr int kVocabSize = 128;

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override { return kVocabSize; }
  int bos_id() const override { return 1; }
  int eos_id() const override { return 2; }
  int unk_id() const override { return 0; }
};

// ---------------------------------------------------------------------------
// JSON 词表分词器。
// 文件格式：[{"id": 0, "content": "<unk>", "special": true}, ...]
// 编码时从当前位置找「最长能对上的词」，对不上就记成 UNK。
// ---------------------------------------------------------------------------
class JsonVocabTokenizer : public ITokenizer {
 public:
  explicit JsonVocabTokenizer(const std::string& vocab_path);

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override { return vocab_size_; }
  int bos_id() const override { return bos_id_; }
  int eos_id() const override { return eos_id_; }
  int unk_id() const override { return unk_id_; }

 private:
  struct VocabEntry {
    int id = 0;
    std::string content;
    bool special = false;
  };

  int vocab_size_ = 0;
  int bos_id_ = 1;
  int eos_id_ = 2;
  int unk_id_ = 0;
  std::vector<VocabEntry> id_to_entry_;
  // 按字符串长度从长到短排，保证先匹配 "hello" 再匹配 "h"
  std::vector<std::pair<std::string, int>> content_to_id_;
};

// ---------------------------------------------------------------------------
// BPE 分词器（GPT-2 风格）。
// 需要三份数据：
//   vocab.json           片段 -> id
//   merges.txt           允许合并的相邻对，越靠前优先级越高
//   special_tokens.json  bos / eos / unk 的 id
// ---------------------------------------------------------------------------
class BpeTokenizer : public ITokenizer {
 public:
  // 读入三份文件。失败返回 false。
  bool Load(const std::string& vocab_path, const std::string& merges_path,
            const std::string& special_path);

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
  // (左片段, 右片段) -> 优先级，数字越小越先合并
  std::map<std::pair<std::string, std::string>, int> merge_ranks_;
  // 特殊串（如 <|im_start|>），按长度从长到短，编码时优先整段吃掉
  std::vector<std::pair<std::string, int>> special_tokens_;
  std::vector<std::string> b2u_;
  std::unordered_map<std::string, uint8_t> u2b_;
  int bos_id_ = -1;
  int eos_id_ = -1;
  int unk_id_ = -1;
};

// 词表文件存在就用 JSON 词表，否则退回 ASCII。
std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path);

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path,
                                               const std::string& merges_path,
                                               const std::string& special_path);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_TOKENIZER_H_
