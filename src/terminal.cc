//终端交互：读用户输入，打出助手回复，打印/help/stats等
#include "mini_llama/terminal.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace mini_llama {

#if defined(_WIN32)
namespace {

bool IsConsoleHandle(HANDLE handle) {
  DWORD mode = 0;
  return handle != INVALID_HANDLE_VALUE && GetConsoleMode(handle, &mode) != 0;
}
//宽字符转UTF-8
std::string WideToUtf8(const std::wstring& wide) {
  if (wide.empty()) {
    return "";
  }
  //转换
  int size = WideCharToMultiByte(CP_UTF8, 0, wide.data(),
                                 static_cast<int>(wide.size()), nullptr, 0,
                                 nullptr, nullptr);
  if (size <= 0) {
    return "";
  }
  std::string out(static_cast<size_t>(size), '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
                      &out[0], size, nullptr, nullptr);
  return out;
}
//UTF-8转宽字符
std::wstring Utf8ToWide(const std::string& utf8) {
  if (utf8.empty()) {
    return L"";
  }
  //转换
  int size = MultiByteToWideChar(CP_UTF8, 0, utf8.data(),
                                 static_cast<int>(utf8.size()), nullptr, 0);
  if (size <= 0) {
    return L"";
  }
  std::wstring out(static_cast<size_t>(size), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()),
                      &out[0], size);
  return out;
}
//写UTF-8到控制台
void WriteUtf8ToConsole(HANDLE handle, const std::string& text) {
  //判断是否是控制台句柄
  if (!IsConsoleHandle(handle)) {
    //不是控制台句柄，直接输出
    std::cout << text;
    return;
  }
  std::wstring wide = Utf8ToWide(text);
  if (wide.empty()) {
    return;
  }
  DWORD written = 0;
  WriteConsoleW(handle, wide.data(), static_cast<DWORD>(wide.size()), &written,
                nullptr);
}

}
#endif

//初始化终端UTF-8编码
void InitTerminalUtf8() {
#if defined(_WIN32)
  SetConsoleCP(CP_UTF8);
  SetConsoleOutputCP(CP_UTF8);
#endif
}

//打印用户提示符
void Terminal::PrintUserPrompt() const {
#if defined(_WIN32)
  WriteUtf8ToConsole(GetStdHandle(STD_OUTPUT_HANDLE), "mini-ai > ");
#else
  std::cout << "mini-ai > ";
  std::cout.flush();
#endif
}

//读一行
std::string Terminal::ReadLine() const {
#if defined(_WIN32)
  HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
  //判断是否是控制台句柄
  if (IsConsoleHandle(input)) {
    //读取控制台输入
    wchar_t buffer[4096];
    DWORD read = 0;
    if (!ReadConsoleW(input, buffer, 4095, &read, nullptr)) {
      return "";
    }
    //去掉末尾的换行符
    while (read > 0 &&
           (buffer[read - 1] == L'\n' || buffer[read - 1] == L'\r')) {
      --read;
    }
    //转换为UTF-8
    return WideToUtf8(std::wstring(buffer, buffer + read));
  }
#endif
  std::string line;
  if (std::getline(std::cin, line)) {
    return line;
  }
  return "";
}

//打印助手提示符
void Terminal::PrintAssistantPrefix() const {
#if defined(_WIN32)
  WriteUtf8ToConsole(GetStdHandle(STD_OUTPUT_HANDLE), "assistant  > ");
#else
  std::cout << "assistant  > ";
  std::cout.flush();
#endif
}

//打印token文本
void Terminal::PrintTokenText(const std::string& text) const {
#if defined(_WIN32)
  WriteUtf8ToConsole(GetStdHandle(STD_OUTPUT_HANDLE), text);
#else
  std::cout << text;
#endif
}

//刷新输出
void Terminal::Flush() const { std::cout.flush(); }

//打印换行
void Terminal::NewLine() const {
#if defined(_WIN32)
  WriteUtf8ToConsole(GetStdHandle(STD_OUTPUT_HANDLE), "\n");
#else
  std::cout << "\n";
#endif
}

//打印帮助
void Terminal::PrintHelp() const {
  PrintMessage(
      "Commands:\n"
      "  /help    Show this help message\n"
      "  /clear   Clear chat history and context\n"
      "  /stats   Show session statistics\n"
      "  /params  Show current sampling parameters\n"
      "  /exit    Exit chat");
}

//打印会话统计
void Terminal::PrintStats(const ChatSession& session) const {
  std::ostringstream oss;
  oss << "Session stats:\n"
      << "  messages:            " << session.messages.size() << "\n"
      << "  context tokens:      " << session.token_history.size() << "\n"
      << "  total prompt tokens: " << session.total_prompt_tokens << "\n"
      << "  total generated:     " << session.total_generated_tokens << "\n"
      << "  total time:          " << std::fixed << std::setprecision(2)
      << session.total_time_ms << " ms\n"
      << "  tokens/s:            ";
  if (session.total_time_ms > 0.0) {
    oss << std::fixed << std::setprecision(2)
        << (session.total_generated_tokens * 1000.0 / session.total_time_ms);
  } else {
    oss << "n/a";
  }
  PrintMessage(oss.str());
}

//打印采样参数
void Terminal::PrintParams(const SamplingParams& params) const {
  std::ostringstream oss;
  oss << std::defaultfloat << "Sampling params:\n"
      << "  temperature: " << params.temperature << "\n"
      << "  top_k:       " << params.top_k << "\n"
      << "  seed:        " << params.seed;
  PrintMessage(oss.str());
}

//打印消息
void Terminal::PrintMessage(const std::string& msg) const {
#if defined(_WIN32)
  WriteUtf8ToConsole(GetStdHandle(STD_OUTPUT_HANDLE), msg);
  WriteUtf8ToConsole(GetStdHandle(STD_OUTPUT_HANDLE), "\n");
#else
  std::cout << msg << "\n";
#endif
}

}
