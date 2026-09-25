#ifndef INCLUDE_MINI_LLAMA_CHAT_H_
#define INCLUDE_MINI_LLAMA_CHAT_H_

#include <string>
#include <vector>

#include "mini_llama/radix_tree.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

struct ChatMessage {
  std::string role;
  std::string content;
};

//一次聊天会话的状态
class ChatSession {
 public:
  std::vector<ChatMessage> messages;
  std::vector<int> token_history;
  RadixTree prefix_cache;//已经算过的token前缀
  SamplingParams sampling_params;

  int total_prompt_tokens = 0;
  int total_generated_tokens = 0;
  double total_time_ms = 0.0;

  ChatSession() = default;

  void Clear();//清空消息，历史，前缀缓存，统计
  void AddMessage(const std::string& role, const std::string& content);
  void RecordTurn(int prompt_tokens, int generated_tokens, double time_ms);
  void SetTokenHistory(const std::vector<int>& tokens);
  void AppendToken(int token);
  void RecordPrefix(const std::vector<int>& tokens);//记到前缀树
  size_t LongestCachedPrefix(const std::vector<int>& tokens) const;
};

}

#endif
