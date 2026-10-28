#include "mini_llama/prompt_builder.h"
#include "tests/test_main.h"
#include <stdexcept>

using namespace mini_llama;

static bool TestQwenJinja() {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{% for message in messages %}"
      "{% if loop.first and messages[0]['role'] != 'system' %}"
      "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
      "{% endif %}"
      "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + "
      "'<|im_end|>\\n' }}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif "
      "%}");
  PromptBuilder builtin;
  builtin.SetChatTemplate("qwen2");
  const std::vector<ChatMessage> messages{
      {"user", "你好"}, {"assistant", "Hello"}, {"user", "继续"}};
  MINI_LLAMA_ASSERT_EQ(builder.Build(messages), builtin.Build(messages));
  builder.SetChatTemplate("prefix  {{- \"ok\\n\" -}}  suffix");
  MINI_LLAMA_ASSERT_EQ(builder.Build(messages),
                       std::string("prefixok\nsuffix"));
  return true;
}

static bool TestRejectsUnsupportedJinja() {
  const std::vector<ChatMessage> messages{{"user", "hello"}};
  for (const char *input : {"{% set x = 1 %}", "{% for message in messages %}",
                            "{{ missing_variable }}"}) {
    PromptBuilder builder;
    builder.SetChatTemplate(input);
    bool rejected = false;
    try {
      builder.Build(messages);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    MINI_LLAMA_ASSERT_TRUE(rejected);
  }
  return true;
}

static struct TemplateRegistrar {
  TemplateRegistrar() {
    RegisterTest("qwen_jinja_matches_builtin", TestQwenJinja);
    RegisterTest("unsupported_jinja_is_rejected", TestRejectsUnsupportedJinja);
  }
} template_registrar;
