#include "mini_llama/radix_tree.h"

#include <utility>

namespace mini_llama {

RadixTree::RadixTree() = default;
RadixTree::~RadixTree() = default;
RadixTree::RadixTree(RadixTree&&) noexcept = default;
RadixTree& RadixTree::operator=(RadixTree&&) noexcept = default;

//清空整棵树
void RadixTree::Clear() {
  root_.children.clear();
  root_.terminal = false;
  root_.key.clear();
  terminal_count_ = 0;
}

bool RadixTree::empty() const { return terminal_count_ == 0; }

size_t RadixTree::size() const { return terminal_count_; }

//从a[a_offset..]和b[0..] 比，看公共前缀多长
size_t RadixTree::CommonPrefixLength(const std::vector<int>& a,
                                     size_t a_offset,
                                     const std::vector<int>& b) {
  size_t i = 0;
  while (a_offset + i < a.size() && i < b.size() &&
         a[a_offset + i] == b[i]) {
    ++i;
  }
  return i;
}

//插入一条token序列
void RadixTree::Insert(const std::vector<int>& tokens) {
  if (tokens.empty()) {
    //空序列也算一条，标在根上
    if (!root_.terminal) {
      root_.terminal = true;
      ++terminal_count_;
    }
    return;
  }
  InsertInto(&root_, tokens);
}

//往parent下面插suffix
void RadixTree::InsertInto(Node* parent, std::vector<int> suffix) {
  auto child_it = parent->children.find(suffix[0]);
  if (child_it == parent->children.end()) {
    //没有同首 token 的孩子，整段直接挂上去
    auto child = std::make_unique<Node>();
    child->key = std::move(suffix);
    child->terminal = true;
    parent->children[child->key[0]] = std::move(child);
    ++terminal_count_;
    return;
  }

  std::unique_ptr<Node>& child = child_it->second;
  size_t common = CommonPrefixLength(suffix, 0, child->key);
  if (common == child->key.size()) {
    //旧边被新序列完整吃掉
    if (common == suffix.size()) {
      //两条一样，别重复计数
      if (!child->terminal) {
        child->terminal = true;
        ++terminal_count_;
      }
      return;
    }
    //还有剩余，往更深处插
    std::vector<int> remaining(suffix.begin() + static_cast<long>(common),
                               suffix.end());
    InsertInto(child.get(), std::move(remaining));
    return;
  }

  //分叉：公共前缀拆成中间节点
  auto split = std::make_unique<Node>();
  split->key.assign(child->key.begin(),
                    child->key.begin() + static_cast<long>(common));

  //旧孩子剩下的那段挂到 split 下面
  child->key.erase(child->key.begin(),
                   child->key.begin() + static_cast<long>(common));
  int old_child_first = child->key[0];
  split->children[old_child_first] = std::move(child);

  if (common == suffix.size()) {
    //新序列刚好停在分叉点
    split->terminal = true;
    ++terminal_count_;
  } else {
    //新序列还有尾巴，再挂一个孩子
    auto new_child = std::make_unique<Node>();
    new_child->key.assign(suffix.begin() + static_cast<long>(common),
                          suffix.end());
    new_child->terminal = true;
    split->children[new_child->key[0]] = std::move(new_child);
    ++terminal_count_;
  }

  child_it->second = std::move(split);
}

//查询最长已缓存前缀长度
size_t RadixTree::LongestPrefix(const std::vector<int>& tokens) const {
  if (tokens.empty()) {
    return 0;
  }

  const Node* node = &root_;
  size_t pos = 0;
  while (pos < tokens.size()) {
    auto child_it = node->children.find(tokens[pos]);
    if (child_it == node->children.end()) {
      return pos;//对不上了，返回已经匹配的长度
    }

    const Node* child = child_it->second.get();
    size_t common = CommonPrefixLength(tokens, pos, child->key);
    pos += common;
    if (common < child->key.size()) {
      return pos;//边没走完就分叉了
    }
    node = child;
  }
  return pos;
}

}
