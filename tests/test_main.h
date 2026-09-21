#ifndef TESTS_TEST_MAIN_H_
#define TESTS_TEST_MAIN_H_

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

//一个测试用例=名字+返回bool的函数，true表示通过
struct TestCase {
  std::string name;
  std::function<bool()> fn;
};

std::vector<TestCase>& TestRegistry();
void RegisterTest(const std::string& name, std::function<bool()> fn);

inline bool AssertTrue(bool cond, const char* expr, const char* file,
                       int line) {
  if (!cond) {
    std::cerr << "  ASSERTION FAILED: " << expr << " at " << file << ":" << line
              << std::endl;
    return false;
  }
  return true;
}

inline bool AssertEqFloat(float a, float b, float tol, const char* expr_a,
                          const char* expr_b, const char* file, int line) {
  if (std::fabs(a - b) > tol) {
    std::cerr << "  ASSERTION FAILED: |" << expr_a << " - " << expr_b << "| = |"
              << a << " - " << b << "| = " << std::fabs(a - b) << " > " << tol
              << " at " << file << ":" << line << std::endl;
    return false;
  }
  return true;
}

//条件为真，否则测试失败
#define MINI_LLAMA_ASSERT_TRUE(cond)                      \
  do {                                                    \
    if (!AssertTrue((cond), #cond, __FILE__, __LINE__)) { \
      return false;                                       \
    }                                                     \
  } while (0)

//精确相等，适合整数和shape
#define MINI_LLAMA_ASSERT_EQ(a, b)                                   \
  do {                                                               \
    if (!AssertTrue((a) == (b), #a " == " #b, __FILE__, __LINE__)) { \
      return false;                                                  \
    }                                                                \
  } while (0)

//浮点近似相等，允许一点点误差
#define MINI_LLAMA_ASSERT_NEAR(a, b, tol)                              \
  do {                                                                 \
    if (!AssertEqFloat((a), (b), (tol), #a, #b, __FILE__, __LINE__)) { \
      return false;                                                    \
    }                                                                  \
  } while (0)

//断言失败，打印消息和位置
#define MINI_LLAMA_ASSERT_FAIL(msg)                                         \
  do {                                                                      \
    std::cerr << "  ASSERTION FAILED: " << msg << " at " << __FILE__ << ":" \
              << __LINE__ << std::endl;                                     \
    return false;                                                           \
  } while (0)

#endif  
