#include "mini_llama/chat.h"

namespace mini_llama {

//清空会话状态
void ChatSession::Clear() {
  messages.clear();
  token_history.clear();
  prefix_cache.Clear();
  total_prompt_tokens = 0;
  total_generated_tokens = 0;
  total_time_ms = 0.0;
}

//追加一条消息
void ChatSession::AddMessage(const std::string& role,
                             const std::string& content) {
  messages.push_back({role, content});
}

//累加一轮的token数和耗时
void ChatSession::RecordTurn(int prompt_tokens, int generated_tokens,
                             double time_ms) {
  total_prompt_tokens += prompt_tokens;
  total_generated_tokens += generated_tokens;
  total_time_ms += time_ms;
}

//整段替换token历史
void ChatSession::SetTokenHistory(const std::vector<int>& tokens) {
  token_history = tokens;
}

//追加一个生成出来的token
void ChatSession::AppendToken(int token) { token_history.push_back(token); }

//把已经算过的前缀记到树里，下次可以复用
void ChatSession::RecordPrefix(const std::vector<int>& tokens) {
  prefix_cache.Insert(tokens);
}

//记录这条序列最长能命中多长的缓存前缀长度
size_t ChatSession::LongestCachedPrefix(const std::vector<int>& tokens) const {
  return prefix_cache.LongestPrefix(tokens);
}

}
