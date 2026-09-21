//极简测试的入口：把注册好的用例一个个跑，统计通过和失败
#include "tests/test_main.h"

#include <iostream>
#include <string>
#include <vector>

std::vector<TestCase>& TestRegistry() {
  static std::vector<TestCase> registry;//只建一次，后面注册都往这里塞
  return registry;
}

void RegisterTest(const std::string& name, std::function<bool()> fn) {
  TestRegistry().push_back({name, fn});//登记一个用例
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  const auto& tests = TestRegistry();
  int passed = 0;
  int failed = 0;

  std::cout << "Running " << tests.size() << " tests..." << std::endl;
  std::cout << "========================================" << std::endl;

  for (const auto& t : tests) {
    std::cout << "[RUN] " << t.name << std::endl;
    const bool ok = t.fn();//true算通过
    if (ok) {
      std::cout << "[PASS] " << t.name << std::endl;
      ++passed;
    } else {
      std::cout << "[FAIL] " << t.name << std::endl;
      ++failed;
    }
  }

  std::cout << "========================================" << std::endl;
  std::cout << "Results: " << passed << " passed, " << failed << " failed"
            << std::endl;
  return failed > 0 ? 1 : 0;//有失败就让程序返回非0
}
