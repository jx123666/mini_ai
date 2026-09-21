//从JSON词表加载"字符串片段->id"，编码时做最长匹配
//和ASCII的差别：ASCII永远一个字节一个id。JSON词表可以有多字符词，比如同时有"h"和"hello"时，编码"hello"会整段匹配成一个id，而不是五个字母
#include "mini_llama/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {
namespace {
//解析JSON字符串
std::string ParseJsonString(const std::string& content, size_t& pos) {
  //遍历content，如果pos小于content的大小且content[pos]不为"，则将pos加1
  while (pos < content.size() && content[pos] != '"') {
    ++pos;
  }
  if (pos >= content.size()) {//如果pos大于等于content的大小，异常退出
    throw std::runtime_error("expected string");
  }
  ++pos;//跳过开头的引号
  std::string result;
  //遍历content，如果pos小于content的大小且content[pos]不为"，则继续遍历
  while (pos < content.size() && content[pos] != '"') {
    //如果content[pos]为'\'且pos+1小于content的大小，则把转义还原成真正的字符
    if (content[pos] == '\\' && pos + 1 < content.size()) {
      const char next = content[pos + 1];
      if (next == 'n') {
        result += '\n';//\n还原成换行
      } else if (next == 't') {
        result += '\t';//\t还原成制表符
      } else if (next == 'r') {
        result += '\r';//\r还原成回车
      } else if (next == '\\' || next == '"' || next == '/') {
        result += next;//\\ \" \/还原成原字符
      } else {
        result += next;//不认识的转义也原样收下
      }
      pos += 2;//转义占两个字符，指针往后跳两位
    } else {
      result += content[pos];//普通字符直接收下
      ++pos;
    }
  }
  if (pos >= content.size()) {//走到头还没遇到收尾引号
    throw std::runtime_error("unterminated string");
  }
  ++pos;//跳过结尾的引号
  return result;
}
//解析JSON整数
int ParseJsonInt(const std::string& content, size_t& pos) {
  //遍历content，如果pos小于content的大小且content[pos]不为数字且content[pos]不为'-'，则将pos加1
  while (pos < content.size() &&
         !std::isdigit(static_cast<unsigned char>(content[pos])) &&
         content[pos] != '-') {
    ++pos;
  }
  const size_t start = pos;//记下数字开始的位置
  if (pos < content.size() && content[pos] == '-') {
    ++pos;//负数，先跳过负号
  }
  while (pos < content.size() &&
         std::isdigit(static_cast<unsigned char>(content[pos]))) {
    ++pos;//把后面的数字都扫完
  }
  //将content的start到pos-1的子串转换为整数
  return std::stoi(content.substr(start, pos - start));
}

//解析JSON布尔值
bool ParseJsonBool(const std::string& content, size_t& pos) {
  if (content.compare(pos, 4, "true") == 0) {
    pos += 4;//匹配到true，指针跳过这4个字符
    return true;
  }
  if (content.compare(pos, 5, "false") == 0) {
    pos += 5;//匹配到false，指针跳过这5个字符
    return false;
  }
  throw std::runtime_error("expected true or false");//既不是true也不是false
}

}  
//加载JSON词表
JsonVocabTokenizer::JsonVocabTokenizer(const std::string& vocab_path) {
  std::ifstream file(vocab_path);//打开vocab_path
  if (!file.is_open()) {
    throw std::runtime_error("failed to open vocab: " + vocab_path);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();//将文件内容读入buffer
  const std::string content = buffer.str();
  //在content中查找"["
  size_t pos = content.find('[');
  if (pos == std::string::npos) {
    throw std::runtime_error("vocab JSON must be an array");
  }
  ++pos;
  //遍历content，如果pos小于content的大小且content[pos]不为'{'，则将pos加1
  int max_id = -1;
  while (pos < content.size()) {
    while (pos < content.size() && content[pos] != '{') {
      //如果content[pos]为']'，则将pos设置为content的大小
      if (content[pos] == ']') {
        pos = content.size();
        break;
      }
      ++pos;
    }
    if (pos >= content.size() || content[pos] != '{') {
      break;
    }
    ++pos;//跳过'{'
    //解析词表项
    VocabEntry entry;
    bool has_id = false;
    bool has_content = false;
    while (pos < content.size() && content[pos] != '}') {
      while (pos < content.size() &&
             (std::isspace(static_cast<unsigned char>(content[pos])) ||
              content[pos] == ',')) {
        ++pos;
      }
      if (pos >= content.size() || content[pos] == '}') {
        break;
      }
      const std::string key = ParseJsonString(content, pos);//读字段名，比如"id"
      while (pos < content.size() && content[pos] != ':') {
        ++pos;
      }
      if (pos < content.size()) {
        ++pos;
      }
      if (key == "id") {
        entry.id = ParseJsonInt(content, pos);//记下编号
        has_id = true;
      } else if (key == "content") {
        entry.content = ParseJsonString(content, pos);//记下这段文字
        has_content = true;
      } else if (key == "special") {
        entry.special = ParseJsonBool(content, pos);//是不是特殊符号
      } else if (pos < content.size() && content[pos] == '"') {
        ParseJsonString(content, pos);//不认识的字符串字段，读掉丢掉
      } else {
        while (pos < content.size() && content[pos] != ',' &&
               content[pos] != '}') {
          ++pos;//不认识的其它值，跳到逗号或右花括号
        }
      }
    }
    if (pos < content.size() && content[pos] == '}') {
      ++pos;//跳过'}'
    }
    if (!has_id || !has_content) {
      throw std::runtime_error("vocab entry missing id or content");//这条词没有id或content
    }
    if (entry.content == "<bos>") {
      bos_id_ = entry.id;//句子开头的编号
    } else if (entry.content == "<eos>") {
      eos_id_ = entry.id;//句子结束的编号
    } else if (entry.content == "<unk>") {
      unk_id_ = entry.id;//不认识的字用的编号
    }
    if (entry.id > max_id) {
      max_id = entry.id;//记下目前最大的id
    }
    if (static_cast<size_t>(entry.id) >= id_to_entry_.size()) {
      id_to_entry_.resize(static_cast<size_t>(entry.id) + 1);//id可能不连续，把表扩到够放下这个id
    }
    id_to_entry_[entry.id] = entry;//用id查内容，解码用
    content_to_id_.push_back({entry.content, entry.id});//用内容查id，编码用
  }

  if (max_id < 0) {
    throw std::runtime_error("vocab file contains no entries");
  }
  vocab_size_ = max_id + 1;
  //按字符串从长到短排，编码时先匹配"hello"再匹配"h"
  std::sort(content_to_id_.begin(), content_to_id_.end(),
            [](const auto& a, const auto& b) {
              return a.first.size() > b.first.size();
            });
}

//编码：先放BOS，再从左到右做最长匹配
std::vector<int> JsonVocabTokenizer::Encode(const std::string& text) const {
  std::vector<int> tokens;
  tokens.push_back(bos_id_);//和ASCII一样，最前面加句子开头

  size_t pos = 0;//当前从文本的第几个字符开始看
  while (pos < text.size()) {
    bool matched = false;
    //content_to_id_已经按长度从长到短排过，先碰到的就是最长词
    for (const auto& pair : content_to_id_) {
      const std::string& piece = pair.first;
      if (piece.empty()) {
        continue;//空字符串不能拿来匹配，否则会一直停在原地
      }
      if (text.compare(pos, piece.size(), piece) == 0) {
        tokens.push_back(pair.second);//对上了，记下这个id
        pos += piece.size();//指针跳过这一整段
        matched = true;
        break;
      }
    }
    if (!matched) {
      tokens.push_back(unk_id_);//一个字都对不上，记成UNK
      ++pos;//只吃掉这一个字节，继续往后
    }
  }
  return tokens;
}

//把一个id变回它在词表里的字符串
std::string JsonVocabTokenizer::DecodeToken(int token) const {
  if (token < 0 || static_cast<size_t>(token) >= id_to_entry_.size()) {
    return "<unk>";//id越界，当不认识
  }
  return id_to_entry_[token].content;
}

//把一串id拼回字符串
std::string JsonVocabTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string result;
  for (int token : tokens) {
    result += DecodeToken(token);
  }
  return result;
}
//有词表文件就用JSON分词，打不开或解析失败就退回ASCII
std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path) {
  if (!vocab_path.empty()) {
    std::ifstream file(vocab_path);
    if (file.good()) {
      try {
        return std::make_unique<JsonVocabTokenizer>(vocab_path);
      } catch (const std::exception& e) {
        //词表坏了，打印警告后改用ASCII
        std::cerr << "Warning: failed to load vocab '" << vocab_path
                  << "', falling back to ASCII tokenizer: " << e.what()
                  << std::endl;
      }
    }
  }
  return std::make_unique<AsciiTokenizer>();//没给路径、文件不存在、或解析失败
}

}  
