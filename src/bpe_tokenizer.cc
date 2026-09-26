//从vocab.json + merges.txt + special_tokens.json加载BPE分词器
//合并循环在bpe_common.h的ApplyBpeMerges里

#include "mini_llama/tokenizer.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "src/bpe_common.h"

namespace mini_llama {
namespace {
//解析JSON字符串
bool ParseJsonString(const std::string& s, size_t& pos, std::string& out) {
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
    ++pos;
  }
  if (pos >= s.size() || s[pos] != '"') {
    return false;
  }
  ++pos;
  out.clear();
  //遍历s，如果pos小于s的大小且s[pos]不等于"，则将s[pos]添加到out中
  while (pos < s.size() && s[pos] != '"') {
    if (s[pos] == '\\' && pos + 1 < s.size()) {
      const char esc = s[pos + 1];
      if (esc == 'n') {
        out += '\n';
      } else if (esc == 't') {
        out += '\t';
      } else if (esc == 'r') {
        out += '\r';
      } else if (esc == '"' || esc == '\\' || esc == '/') {
        out += esc;
      } else if (esc == 'u') {
        //词表里常见 \u0120 这种写法，表示一个 Unicode 码点（空格常写成 Ġ）
        //以前只收下了字面量 u0120，空格就对不上词表，会变成 unk
        if (pos + 5 < s.size()) {
          const std::string hex = s.substr(pos + 2, 4);
          try {
            const char32_t cp =
                static_cast<char32_t>(std::stoul(hex, nullptr, 16));
            out += CodepointToUtf8(cp);
          } catch (...) {
            out += '?';
          }
          pos += 4;//多跳过后面 4 位十六进制，下面再 +2 一共跳过 \uXXXX
        }
      } else {
        out += esc;
      }
      pos += 2;
    } else {
      out += s[pos];
      ++pos;
    }
  }
  if (pos < s.size() && s[pos] == '"') {
    ++pos;
  }
  return true;
}

//解析JSON整数
bool ParseJsonInt(const std::string& s, size_t& pos, int& out) {
  //遍历s，如果pos小于s的大小且s[pos]为空格，则将pos加1
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
    ++pos;
  }
  if (pos >= s.size()) {//如果pos大于等于s的大小，则返回false
    return false;
  }
  const size_t start = pos;//start记录当前处理到的位置
  if (s[pos] == '-') {
    ++pos;
  }
  //遍历s，如果pos小于s的大小且s[pos]为数字，则将pos加1
  while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
    ++pos;
  }
  if (start == pos || (s[start] == '-' && pos == start + 1)) {
    return false;
  }
  //将s的start到pos-1的子串转换为整数
  try {
    out = std::stoi(s.substr(start, pos - start));
    return true;
  } catch (...) {
    return false;
  }
}
//查找JSON整数
int FindJsonInt(const std::string& content, const std::string& key) {
  //在content中查找key
  const size_t key_pos = content.find('"' + key + '"');

  if (key_pos == std::string::npos) {
    return -1;
  }
  //在content中查找":"
  const size_t colon = content.find(':', key_pos);
  if (colon == std::string::npos) {
    return -1;
  }
  size_t pos = colon + 1;
  int value = -1;
  if (!ParseJsonInt(content, pos, value)) {
    return -1;
  }
  return value;
}

}  
//加载BPE分词器
bool BpeTokenizer::Load(const std::string& vocab_path,
                        const std::string& merges_path,
                        const std::string& special_path) {
  b2u_ = BuildBytesToUnicode();
  u2b_.clear();
  for (int b = 0; b < 256; ++b) {
    u2b_[b2u_[b]] = static_cast<uint8_t>(b);
  }
  //加载vocab.json
  {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
      std::cerr << "Failed to open vocab: " << vocab_path << std::endl;
      return false;
    }
    std::stringstream buffer;
    //将文件内容读入buffer
    buffer << file.rdbuf();
    const std::string content = buffer.str();
    //在content中查找"{"
    size_t pos = content.find('{');
    if (pos == std::string::npos) {
      return false;
    }
    ++pos;
    //遍历content，如果pos小于content的大小且content[pos]不为'}'，则继续遍历
    while (pos < content.size()) {
      while (pos < content.size() &&
             (std::isspace(static_cast<unsigned char>(content[pos])) ||
              content[pos] == ',')) {
        ++pos;
      }
      if (pos >= content.size() || content[pos] == '}') {
        break;
      }
      std::string key;//key记录当前处理的键
      if (!ParseJsonString(content, pos, key)) {
        break;
      }
      while (pos < content.size() && content[pos] != ':') {
        ++pos;
      }
      if (pos >= content.size()) {
        break;
      }
      ++pos;
      int value = 0;
      if (!ParseJsonInt(content, pos, value)) {
        break;
      }
      vocab_[key] = value;
    }
    if (vocab_.empty()) {
      return false;
    }
    //找到vocab中最大的id
    int max_id = 0;
    for (const auto& kv : vocab_) {
      if (kv.second > max_id) {
        max_id = kv.second;
      }
    }
    id_to_token_.assign(static_cast<size_t>(max_id) + 1, "");
    //将id_to_token_赋值为vocab中最大的id+1
    for (const auto& kv : vocab_) {
      id_to_token_[kv.second] = kv.first;
    }
    //遍历vocab，如果token的长度大于等于4且token的前两个字符为"<|"且token的倒数两个字符为"|>"，则将token和id添加到special_tokens_中
    for (const auto& kv : vocab_) {
      const std::string& token = kv.first;
      if (token.size() >= 4 && token.front() == '<' && token[1] == '|' &&
          token[token.size() - 2] == '|' && token.back() == '>') {
        special_tokens_.push_back({token, kv.second});
      }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto& a, const auto& b) {
                return a.first.size() > b.first.size();
              });
  }
  //加载merges.txt
  {
    std::ifstream file(merges_path);
    if (!file.is_open()) {
      std::cerr << "Failed to open merges: " << merges_path << std::endl;
      return false;
    }
    std::string line;
    int rank = 0;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string left;//left记录当前处理的左边的字符串
      std::string right;//right记录当前处理的右边的字符串
      //如果ss能读到left和right，则将left和right添加到merge_ranks_中
      if (ss >> left >> right) {
        merge_ranks_[{left, right}] = rank++;
      }
    }
  }

  {
    std::ifstream file(special_path);
    if (file.is_open()) {
      std::stringstream buffer;
      buffer << file.rdbuf();
      const std::string content = buffer.str();
      bos_id_ = FindJsonInt(content, "bos_id");
      eos_id_ = FindJsonInt(content, "eos_id");
      unk_id_ = FindJsonInt(content, "unk_id");
    }
  }
  return true;
}
//BPE编码
std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
  return EncodeWithBpe(text, b2u_, vocab_, merge_ranks_, special_tokens_,
                       unk_id_);
}
//解码单个token
std::string BpeTokenizer::DecodeToken(int token) const {
  if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
    return id_to_token_[token];
  }
  return "";
}
//解码多个token
std::string BpeTokenizer::Decode(const std::vector<int>& tokens) const {
  return DecodeWithBpe(tokens, id_to_token_, u2b_);
}
//创建BPE分词器
std::unique_ptr<ITokenizer> CreateBpeTokenizer(
    const std::string& vocab_path, const std::string& merges_path,
    const std::string& special_path) {
  auto tok = std::make_unique<BpeTokenizer>();
  if (!tok->Load(vocab_path, merges_path, special_path)) {
    return nullptr;
  }
  return tok;
}

}
