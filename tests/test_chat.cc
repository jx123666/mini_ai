#include "mini_llama/chat.h"
#include "tests/test_main.h"

using mini_llama::ChatMessage;
using mini_llama::ChatSession;

//消息能存住role和content
static bool TestChatMessageStoresRoleAndContent() {
  ChatMessage msg{"user", "hello"};
  MINI_LLAMA_ASSERT_TRUE(msg.role == "user");
  MINI_LLAMA_ASSERT_TRUE(msg.content == "hello");
  return true;
}

//新建会话应该是空的
static bool TestChatSessionStartsEmpty() {
  ChatSession session;
  MINI_LLAMA_ASSERT_TRUE(session.messages.empty());
  MINI_LLAMA_ASSERT_TRUE(session.token_history.empty());
  MINI_LLAMA_ASSERT_EQ(session.total_prompt_tokens, 0);
  MINI_LLAMA_ASSERT_EQ(session.total_generated_tokens, 0);
  MINI_LLAMA_ASSERT_NEAR(session.total_time_ms, 0.0, 1e-6);
  return true;
}

//AddMessage往后面追加
static bool TestChatSessionAddMessageAppends() {
  ChatSession session;
  session.AddMessage("system", "You are helpful.");
  session.AddMessage("user", "hi");
  MINI_LLAMA_ASSERT_EQ(session.messages.size(), 2);
  MINI_LLAMA_ASSERT_TRUE(session.messages[0].role == "system");
  MINI_LLAMA_ASSERT_TRUE(session.messages[0].content == "You are helpful.");
  MINI_LLAMA_ASSERT_TRUE(session.messages[1].role == "user");
  MINI_LLAMA_ASSERT_TRUE(session.messages[1].content == "hi");
  return true;
}

//Clear全清空
static bool TestChatSessionClearResets() {
  ChatSession session;
  session.AddMessage("user", "test");
  session.SetTokenHistory({1, 2, 3});
  session.RecordPrefix({1, 2, 3});
  session.RecordTurn(10, 5, 100.0);
  session.Clear();
  MINI_LLAMA_ASSERT_TRUE(session.messages.empty());
  MINI_LLAMA_ASSERT_TRUE(session.token_history.empty());
  MINI_LLAMA_ASSERT_EQ(session.LongestCachedPrefix({1, 2, 3}), 0);
  MINI_LLAMA_ASSERT_EQ(session.total_prompt_tokens, 0);
  MINI_LLAMA_ASSERT_EQ(session.total_generated_tokens, 0);
  MINI_LLAMA_ASSERT_NEAR(session.total_time_ms, 0.0, 1e-6);
  return true;
}
//测试token历史
static bool TestChatSessionTracksTokenHistory() {
  ChatSession session;
  session.SetTokenHistory({1, 10, 11});
  session.AppendToken(12);
  MINI_LLAMA_ASSERT_EQ(session.token_history.size(), 4);
  MINI_LLAMA_ASSERT_EQ(session.token_history[0], 1);
  MINI_LLAMA_ASSERT_EQ(session.token_history[3], 12);
  return true;
}

//前缀缓存靠RadixTree，能查最长命中
static bool TestChatSessionRecordsPrefixCache() {
  ChatSession session;
  session.RecordPrefix({1, 2, 3, 4});
  session.RecordPrefix({1, 2, 9});
  MINI_LLAMA_ASSERT_EQ(session.LongestCachedPrefix({1, 2, 3}), 3);
  MINI_LLAMA_ASSERT_EQ(session.LongestCachedPrefix({1, 2, 9, 8}), 3);
  MINI_LLAMA_ASSERT_EQ(session.LongestCachedPrefix({1, 8}), 1);
  return true;
}

//RecordTurn会累加统计
static bool TestChatSessionRecordTurnAccumulates() {
  ChatSession session;
  session.RecordTurn(10, 5, 100.0);
  session.RecordTurn(20, 8, 200.0);
  MINI_LLAMA_ASSERT_EQ(session.total_prompt_tokens, 30);
  MINI_LLAMA_ASSERT_EQ(session.total_generated_tokens, 13);
  MINI_LLAMA_ASSERT_NEAR(session.total_time_ms, 300.0, 1e-6);
  return true;
}
//测试会话
static struct ChatTestRegistrar {
  ChatTestRegistrar() {
    RegisterTest("chat_message_stores_role_and_content",
                 TestChatMessageStoresRoleAndContent);
    RegisterTest("chat_session_starts_empty", TestChatSessionStartsEmpty);
    RegisterTest("chat_session_add_message_appends",
                 TestChatSessionAddMessageAppends);
    RegisterTest("chat_session_clear_resets", TestChatSessionClearResets);
    RegisterTest("chat_session_record_turn_accumulates",
                 TestChatSessionRecordTurnAccumulates);
    RegisterTest("chat_session_tracks_token_history",
                 TestChatSessionTracksTokenHistory);
    RegisterTest("chat_session_records_prefix_cache",
                 TestChatSessionRecordsPrefixCache);
  }
} chat_test_registrar;
