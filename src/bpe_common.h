
#ifndef SRC_BPE_COMMON_H_
#define SRC_BPE_COMMON_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini_llama {
//将Unicode码点转换为UTF-8编码
inline std::string CodepointToUtf8(char32_t cp) {
  std::string result;
  if (cp <= 0x7F) {
    result += static_cast<char>(cp);
  } else if (cp <= 0x7FF) {
    result += static_cast<char>(0xC0 | (cp >> 6));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0xFFFF) {
    result += static_cast<char>(0xE0 | (cp >> 12));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    result += static_cast<char>(0xF0 | (cp >> 18));
    result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return result;
}

//GPT-2的byte->unicode表
//可见字符映射成自己；空格、换行等控制字节映射成别的 Unicode
//这样词表里不会出现真正的空格，文件更好存，解码时再映射回原来的字节
inline std::vector<std::string> BuildBytesToUnicode() {
  std::vector<std::string> map(256);//256个字符的映射表
  std::vector<int> bs;//可见字符的码点
  for (int c = '!'; c <= '~'; ++c) {
    bs.push_back(c);
  }
  for (int c = 0xA1; c <= 0xAC; ++c) {
    bs.push_back(c);
  }
  for (int c = 0xAE; c <= 0xFF; ++c) {
    bs.push_back(c);
  }
  //将bs中的码点转换为Unicode码点
  std::vector<int> cs
  //遍历256个字符，如果bs中没有b，则将b添加到bs中，并将其转换为Unicode码点
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  //将bs中的码点转换为UTF-8编码
  for (size_t i = 0; i < bs.size(); ++i) {
    map[bs[i]] = CodepointToUtf8(static_cast<char32_t>(cs[i]));
  }
  return map;
}
//获取UTF-8编码的下一个码点
inline std::string NextUtf8Codepoint(const std::string& s, size_t& pos) {
  if (pos >= s.size()) {//如果pos大于等于s的大小，则返回空字符串
    return "";
  }
  const unsigned char first = static_cast<unsigned char>(s[pos]);//获取s的第pos个字符
  size_t len = 1;
  if ((first & 0x80) == 0) {//如果第一个字节的高位为0，则len为1
    len = 1;
  } else if ((first & 0xE0) == 0xC0) {//如果第一个字节的高位为110，则len为2
    len = 2;
  } else if ((first & 0xF0) == 0xE0) {
    len = 3;
  } else if ((first & 0xF8) == 0xF0) {
    len = 4;
  } else {
    ++pos;
    return "";
  }
  if (pos + len > s.size()) {
    ++pos;
    return "";
  }
  std::string result = s.substr(pos, len);//从pos开始，截取len个字符
  pos += len;
  return result;
}


//BPE合并
//pieces一开始是一个个字节对应的小片段，例如 hello
//["h", "e", "l", "l", "o"]
//ranks记录哪些相邻对可以合并，数字越小越优先
//每一轮看当前所有相邻对，选rank最小的那一对拼起来，直到不能再合并
inline std::vector<std::string> ApplyBpeMerges(
    std::vector<std::string> pieces,
    const std::map<std::pair<std::string, std::string>, int>& ranks) {
  while (pieces.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_idx = pieces.size();
    for (size_t i = 0; i + 1 < pieces.size(); ++i) {
      const auto it = ranks.find({pieces[i], pieces[i + 1]});
      if (it != ranks.end() && it->second < best_rank) {
        best_rank = it->second;
        best_idx = i;
      }
    }
    if (best_idx >= pieces.size()) {
      break;
    }
    pieces[best_idx] += pieces[best_idx + 1];
    pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best_idx) + 1);
  }
  return pieces;
}
//BPE编码
inline std::vector<int> EncodeWithBpe(
    const std::string& text, const std::vector<std::string>& b2u,
    const std::unordered_map<std::string, int>& vocab,
    const std::map<std::pair<std::string, std::string>, int>& ranks,
    const std::vector<std::pair<std::string, int>>& specials, int unk_id) {
  if (text.empty()) {
    return {};
  }
  //result记录编码结果
  std::vector<int> result;
  size_t pos = 0;//pos记录当前处理到的位置
  while (pos < text.size()) {
    //特殊符号整段匹配，不拆开做BPE
    bool matched = false;
    for (const auto& st : specials) {
      if (pos + st.first.size() <= text.size() &&
          text.compare(pos, st.first.size(), st.first) == 0) {
        result.push_back(st.second);
        pos += st.first.size();
        matched = true;
        break;
      }
    }
    if (matched) {
      continue;
    }

    size_t end = text.size();
    for (const auto& st : specials) {
      const size_t found = text.find(st.first, pos);
      if (found != std::string::npos && found < end) {
        end = found;
      }
    }
    //segment记录当前处理到的片段
    const std::string segment = text.substr(pos, end - pos);
    pos = end;

    std::vector<std::string> pieces;
    pieces.reserve(segment.size());
    for (unsigned char c : segment) {
      pieces.push_back(b2u[c]);
    }
    pieces = ApplyBpeMerges(std::move(pieces), ranks);

    for (const auto& piece : pieces) {
      const auto it = vocab.find(piece);
      if (it != vocab.end()) {
        result.push_back(it->second);
      } else {
        result.push_back(unk_id);
      }
    }
  }
  return result;
}

inline std::string DecodeWithBpe(
    const std::vector<int>& tokens, const std::vector<std::string>& id_to_token,
    const std::unordered_map<std::string, uint8_t>& u2b) {
  std::string text;
  for (int id : tokens) {
    if (id >= 0 && id < static_cast<int>(id_to_token.size())) {
      text += id_to_token[id];
    }
  }
  //text记录解码结果
  //把 GPT-2 的 unicode 符号还原成原始字节。普通字母会原样回来
  std::string result;
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t start = pos;
    const std::string cp = NextUtf8Codepoint(text, pos);
    if (cp.empty()) {
      if (pos == start) {
        ++pos;
      }
      continue;
    }
    const auto it = u2b.find(cp);
    if (it != u2b.end()) {
      result += static_cast<char>(it->second);
    } else {
      result += cp;
    }
  }
  return result;
}

} 

#endif  
