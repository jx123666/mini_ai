//从GGUF metadata装BPE词表：tokens / merges / token_type / bos eos unk
#include "mini_llama/gguf_tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/gguf.h"
#include "src/bpe_common.h"

namespace mini_llama {
namespace {

//ggml里token_type=3表示control，聊天标记通常是这类
constexpr int32_t kTokenTypeControl = 3;
constexpr int32_t kTokenTypeUnknown = 2;

bool LooksLikeSpecial(const std::string& token) {
  return token.size() >= 4 && token.front() == '<' && token[1] == '|' &&
         token[token.size() - 2] == '|' && token.back() == '>';
}

}

void GgufTokenizer::ResetByteMap() {
  b2u_ = BuildBytesToUnicode();
  u2b_.clear();
  for (int b = 0; b < 256; ++b) {
    u2b_[b2u_[static_cast<size_t>(b)]] = static_cast<uint8_t>(b);
  }
}

//建vocab；有token_type就用control当special，否则按<|...|>认
void GgufTokenizer::IndexTokens(const std::vector<std::string>& tokens,
                                const std::vector<int32_t>* token_types) {
  id_to_token_ = tokens;
  vocab_.clear();
  special_tokens_.clear();
  const bool use_types =
      token_types != nullptr && token_types->size() == tokens.size();
  for (size_t i = 0; i < tokens.size(); ++i) {
    vocab_[tokens[i]] = static_cast<int>(i);
    const bool special =
        use_types ? ((*token_types)[i] == kTokenTypeControl)
                  : LooksLikeSpecial(tokens[i]);
    if (special) {
      special_tokens_.push_back({tokens[i], static_cast<int>(i)});
    }
    if (use_types && unk_id_ < 0 &&
        (*token_types)[i] == kTokenTypeUnknown) {
      unk_id_ = static_cast<int>(i);
    }
  }
  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const auto& a, const auto& b) {
              return a.first.size() > b.first.size();
            });
}

void GgufTokenizer::IndexMerges(const std::vector<std::string>& merges) {
  merge_ranks_.clear();
  for (int rank = 0; rank < static_cast<int>(merges.size()); ++rank) {
    const std::string& rule = merges[rank];
    const size_t space = rule.find(' ');
    if (space == std::string::npos || space + 1 >= rule.size()) {
      continue;
    }
    merge_ranks_[{rule.substr(0, space), rule.substr(space + 1)}] = rank;
  }
}

bool GgufTokenizer::Load(const std::string& gguf_path) {
  vocab_.clear();
  id_to_token_.clear();
  merge_ranks_.clear();
  special_tokens_.clear();
  bos_id_ = -1;
  eos_id_ = -1;
  unk_id_ = -1;
  ResetByteMap();

  std::vector<std::string> tokens;
  if (!GgufGetMetadataStringArray(gguf_path, "tokenizer.ggml.tokens",
                                  tokens) ||
      tokens.empty()) {
    std::cerr << "GGUF missing tokenizer.ggml.tokens" << std::endl;
    return false;
  }

  std::vector<int32_t> token_types;
  const bool has_types = GgufGetMetadataIntArray(
      gguf_path, "tokenizer.ggml.token_type", token_types);
  IndexTokens(tokens, has_types ? &token_types : nullptr);

  std::vector<std::string> merges;
  if (GgufGetMetadataStringArray(gguf_path, "tokenizer.ggml.merges", merges)) {
    IndexMerges(merges);
  }

  int64_t bos = -1;
  int64_t eos = -1;
  int64_t unk = -1;
  if (GgufGetMetadataInt(gguf_path, "tokenizer.ggml.bos_token_id", bos)) {
    bos_id_ = static_cast<int>(bos);
  }
  if (GgufGetMetadataInt(gguf_path, "tokenizer.ggml.eos_token_id", eos)) {
    eos_id_ = static_cast<int>(eos);
  }
  //显式unk优先；没有时才用token_type=2扫到的那个
  if (GgufGetMetadataInt(gguf_path, "tokenizer.ggml.unknown_token_id", unk)) {
    unk_id_ = static_cast<int>(unk);
  }

  std::cout << "GGUF tokenizer loaded: vocab=" << vocab_.size()
            << " merges=" << merge_ranks_.size()
            << " special=" << special_tokens_.size() << " bos=" << bos_id_
            << " eos=" << eos_id_ << " unk=" << unk_id_ << std::endl;
  return true;
}

bool GgufTokenizer::LoadPieces(const std::vector<std::string>& tokens,
                               const std::vector<std::string>& merges,
                               int bos_id, int eos_id, int unk_id) {
  bos_id_ = bos_id;
  eos_id_ = eos_id;
  unk_id_ = unk_id;
  ResetByteMap();
  IndexTokens(tokens, nullptr);
  IndexMerges(merges);
  return !tokens.empty();
}

std::vector<int> GgufTokenizer::Encode(const std::string& text) const {
  return EncodeWithBpe(text, b2u_, vocab_, merge_ranks_, special_tokens_,
                       unk_id_);
}

std::string GgufTokenizer::DecodeToken(int token) const {
  if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
    return id_to_token_[static_cast<size_t>(token)];
  }
  return "";
}

std::string GgufTokenizer::Decode(const std::vector<int>& tokens) const {
  return DecodeWithBpe(tokens, id_to_token_, u2b_);
}

std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path) {
  auto tok = std::make_unique<GgufTokenizer>();
  if (!tok->Load(gguf_path)) {
    return nullptr;
  }
  return tok;
}

}
