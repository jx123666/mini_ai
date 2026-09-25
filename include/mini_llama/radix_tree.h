//记哪些token序列算过了，方便复用公共前缀
#ifndef INCLUDE_MINI_LLAMA_RADIX_TREE_H_
#define INCLUDE_MINI_LLAMA_RADIX_TREE_H_

#include <map>
#include <memory>
#include <vector>

namespace mini_llama {

class RadixTree {
 public:
  RadixTree();
  ~RadixTree();

  RadixTree(const RadixTree&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree(RadixTree&&) noexcept;
  RadixTree& operator=(RadixTree&&) noexcept;

  void Clear();
  bool empty() const;
  size_t size() const;//插进去过的完整序列数

  void Insert(const std::vector<int>& tokens);
  //查询这能匹配到多长的已缓存的前缀
  size_t LongestPrefix(const std::vector<int>& tokens) const;

 private:
  struct Node {
    std::vector<int> key;//边上压着的一段token
    bool terminal = false;
    std::map<int, std::unique_ptr<Node>> children;
  };

  //比较公共前缀长度
  static size_t CommonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                   const std::vector<int>& b);
  void InsertInto(Node* parent, std::vector<int> suffix);

  Node root_;
  size_t terminal_count_ = 0;
};

}

#endif
