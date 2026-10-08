//终端交互：读用户输入、打出助手回复，打印/help /stats等
#ifndef INCLUDE_MINI_LLAMA_TERMINAL_H_
#define INCLUDE_MINI_LLAMA_TERMINAL_H_

#include <string>

#include "mini_llama/chat.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

//Windows下把控制台切到UTF-8，中文输入输出才不乱码
void InitTerminalUtf8();

class Terminal {
 public:
  Terminal() = default;

  void PrintUserPrompt() const;//mini-ai 
  std::string ReadLine() const;//读一行，EOF返回空
  void PrintAssistantPrefix() const;//assistant
  void PrintTokenText(const std::string& text) const;//流式打一个片段，不换行
  void Flush() const;
  void NewLine() const;
  void PrintHelp() const;
  void PrintStats(const ChatSession& session) const;
  void PrintParams(const SamplingParams& params) const;
  void PrintMessage(const std::string& msg) const;
};

}

#endif
