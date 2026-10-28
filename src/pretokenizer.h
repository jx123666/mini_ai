#ifndef SRC_PRETOKENIZER_H_
#define SRC_PRETOKENIZER_H_

#include "src/unicode_categories.h"
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace mini_llama {
template <size_t N>
inline bool InUnicodeRanges(char32_t cp, const UnicodeRange (&ranges)[N]) {
  const auto *it = std::lower_bound(
      ranges, ranges + N, cp, [](const UnicodeRange &range, char32_t value) {
        return range.last < value;
      });
  return it != ranges + N && it->first <= cp;
}
inline bool UnicodeSpace(char32_t cp) {
  return (cp >= 9 && cp <= 13) || cp == 0x20 || cp == 0x85 || cp == 0xa0 ||
         cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 ||
         cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

inline std::vector<std::string> SplitPreTokens(const std::string &text,
                                               const std::string &mode) {
  if (mode.empty())
    return {text};
  std::vector<char32_t> cps;
  std::vector<size_t> offsets;
  for (size_t i = 0; i < text.size();) {
    offsets.push_back(i);
    const unsigned char first = static_cast<unsigned char>(text[i++]);
    char32_t cp = first;
    int extra = first < 0x80 ? 0 : first < 0xe0 ? 1 : first < 0xf0 ? 2 : 3;
    if (extra) {
      cp &= (1u << (6 - extra)) - 1;
      for (int j = 0; j < extra && i < text.size(); ++j) {
        const unsigned char next = static_cast<unsigned char>(text[i]);
        if ((next & 0xc0) != 0x80)
          break;
        cp = (cp << 6) | (next & 0x3f);
        ++i;
      }
    }
    cps.push_back(cp);
  }
  offsets.push_back(text.size());
  auto letter = [&](size_t i) {
    return i < cps.size() && InUnicodeRanges(cps[i], kLetterRanges);
  };
  auto number = [&](size_t i) {
    return i < cps.size() && InUnicodeRanges(cps[i], kNumberRanges);
  };
  auto space = [&](size_t i) { return i < cps.size() && UnicodeSpace(cps[i]); };
  auto newline = [&](size_t i) {
    return i < cps.size() && (cps[i] == '\r' || cps[i] == '\n');
  };
  auto punctuation = [&](size_t i) {
    return i < cps.size() && !space(i) && !letter(i) && !number(i);
  };
  const bool qwen = mode == "qwen2";
  std::vector<std::string> result;
  for (size_t begin = 0; begin < cps.size();) {
    size_t end = begin;
    if (cps[begin] == '\'') {
      for (const char *suffix : {"s", "t", "re", "ve", "m", "ll", "d"}) {
        size_t i = begin + 1, j = 0;
        for (; suffix[j] && i < cps.size(); ++i, ++j) {
          char32_t cp = cps[i];
          if (qwen && cp >= 'A' && cp <= 'Z')
            cp += 'a' - 'A';
          if (cp != static_cast<unsigned char>(suffix[j]))
            break;
        }
        if (!suffix[j]) {
          end = i;
          break;
        }
      }
    }
    if (end == begin) {
      size_t word = begin;
      if (qwen ? (!newline(word) && !letter(word) && !number(word))
               : cps[word] == ' ')
        ++word;
      if (letter(word)) {
        end = word + 1;
        while (letter(end))
          ++end;
      }
    }
    if (end == begin) {
      size_t digit = begin;
      if (!qwen && cps[digit] == ' ')
        ++digit;
      if (number(digit)) {
        end = digit + 1;
        while (number(end) && (!qwen || end - digit < 3))
          ++end;
      }
    }
    if (end == begin) {
      size_t punct = begin;
      if (cps[punct] == ' ')
        ++punct;
      if (punctuation(punct)) {
        end = punct + 1;
        while (punctuation(end))
          ++end;
        if (qwen)
          while (newline(end))
            ++end;
      }
    }
    if (end == begin && space(begin)) {
      size_t last_newline = begin;
      end = begin;
      while (space(end)) {
        ++end;
        if (newline(end - 1))
          last_newline = end;
      }
      // 连续空白末尾的一格留给后面的单词，行尾空白则整体保留。
      if (qwen && last_newline > begin)
        end = last_newline;
      else if (end < cps.size() && end - begin > 1)
        --end;
    }
    if (end == begin)
      ++end;
    result.push_back(
        text.substr(offsets[begin], offsets[end] - offsets[begin]));
    begin = end;
  }
  return result;
}
} // namespace mini_llama
#endif
