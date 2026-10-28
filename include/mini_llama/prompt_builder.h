//把多轮聊天消息拼成模型能吃的prompt文本
//默认是System:/User:/Assistant，也可以切到qwen2模板（通义的格式）
//项目要支持多种不同的开源模型，就需要不断增加类似的方法
#ifndef INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_
#define INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_

#include <string>
#include <vector>

#include "mini_llama/chat.h"

namespace mini_llama {

class PromptBuilder {
 public:
  PromptBuilder() = default;
  //设模板名；空字符串走纯文本格式
  void SetChatTemplate(const std::string& template_str);
  //拼出完整prompt，末尾带上assistant前缀，方便模型接着生成
  std::string Build(const std::vector<ChatMessage>& messages) const;

 private:
  std::string chat_template_;

  std::string BuildPlain(const std::vector<ChatMessage>& messages) const;
  std::string BuildQwen2(const std::vector<ChatMessage>& messages) const;
};

//从GGUF读聊天模板；没有模板且架构是qwen2时返回"qwen2"
std::string LoadChatTemplateFromGguf(const std::string& gguf_path);

}

#endif
