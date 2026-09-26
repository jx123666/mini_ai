
#include "mini_llama/gguf_tokenizer.h"
#include <algorithm>
#include <iostream>
#include <utility>
#include "src/bpe_common.h"

namespace mini_llama {
//加载词表和合并规则
bool GgufTokenizer::LoadPieces(const std::vector<std::string>& tokens,
                               const std::vector<std::string>& merges,
                               int bos_id, int eos_id, int unk_id) {
  b2u_ = BuildBytesToUnicode();
  u2b_.clear();
  //遍历256个字符，将字符和其对应的Unicode码点添加到u2b_中
  for (int b = 0; b < 256; ++b) {
    u2b_[b2u_[b]] = static_cast<uint8_t>(b);
  }

  id_to_token_ = tokens;
  vocab_.clear();
  special_tokens_.clear();
  //遍历tokens，将token和其对应的id添加到vocab_中
  for (size_t i = 0; i < tokens.size(); ++i) {
    vocab_[tokens[i]] = static_cast<int>(i);
    const std::string& token = tokens[i];
    if (token.size() >= 4 && token.front() == '<' && token[1] == '|' &&
        token[token.size() - 2] == '|' && token.back() == '>') {
      special_tokens_.push_back({token, static_cast<int>(i)});
    }
  }
  //将special_tokens_按token的长度从大到小排序
  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const auto& a, const auto& b) {
              return a.first.size() > b.first.size();
            });

  merge_ranks_.clear();
  //遍历merges，将merge和其对应的rank添加到merge_ranks_中
  for (int rank = 0; rank < static_cast<int>(merges.size()); ++rank) {
    const std::string& rule = merges[rank];
    const size_t space = rule.find(' ');
    if (space == std::string::npos || space + 1 >= rule.size()) {
      continue;
    }
    //将rule的0到space-1的子串和rule的space+1到rule.size()-1的子串添加到merge_ranks_中
    merge_ranks_[{rule.substr(0, space), rule.substr(space + 1)}] = rank;
  }

  bos_id_ = bos_id;
  eos_id_ = eos_id;
  unk_id_ = unk_id;
  return !tokens.empty();
}
//GGUF编码
std::vector<int> GgufTokenizer::Encode(const std::string& text) const {
  return EncodeWithBpe(text, b2u_, vocab_, merge_ranks_, special_tokens_,
                       unk_id_);
}

//解码单个token
std::string GgufTokenizer::DecodeToken(int token) const {
  if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
    return id_to_token_[token];
  }
  return "";
}
//解码多个token
std::string GgufTokenizer::Decode(const std::vector<int>& tokens) const {
  return DecodeWithBpe(tokens, id_to_token_, u2b_);
}
//创建GGUF分词器
std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path) {
  std::cerr << "CreateGgufTokenizer: GGUF binary loader is not in this "
               "learning chapter ("
            << gguf_path << "). Use GgufTokenizer::LoadPieces instead."
            << std::endl;
  return nullptr;
}

}  
