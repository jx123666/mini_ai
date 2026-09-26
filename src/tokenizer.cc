//ASCII分词器：一个字符一个id，id就是字符自己的编号
#include "mini_llama/tokenizer.h"
#include <string>
#include <vector>

namespace mini_llama {

//编码：先放BOS，再把每个字符的编号按顺序记下
std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
  std::vector<int> tokens;
  tokens.push_back(bos_id());
  for (unsigned char c : text) {
    const int id = static_cast<int>(c);//字符码就是token id
    //词表只有0..127。更高的字节（比如中文的UTF-8）记成UNK
    if (id >= kVocabSize) {
      tokens.push_back(unk_id());
    } else {
      tokens.push_back(id);//可见字符直接用自己的编号
    }
  }
  return tokens;
}

//把一个id变回字符串
std::string AsciiTokenizer::DecodeToken(int token) const {
  if (token == bos_id()) {
    return "<bos>";//句子开头
  }
  if (token == eos_id()) {
    return "<eos>";//句子结束
  }
  if (token == unk_id()) {
    return "<unk>";//不认识的字
  }
  if (token < 0 || token >= kVocabSize) {
    return "<unk>";//超出0..127，也当不认识
  }
  //控制字符（换行、制表符、DEL）不打印成乱码，解码成空串
  //这些字节encode再decode回不来，roundtrip只保证可见字符
  if (token < 32 || token == 127) {
    return "";
  }
  return std::string(1, static_cast<char>(token));//编号还原成那个字符
}

//把一串id拼回字符串
std::string AsciiTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string result;
  for (int token : tokens) {
    result += DecodeToken(token);
  }
  return result;
}

} 
