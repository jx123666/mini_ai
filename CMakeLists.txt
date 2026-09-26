# =============================================================================
# 文件作用：
#   本项目的构建入口。对应学习任务「核心推理模块」中的：
#   1) Tensor 与基础算子
#   2) 单元测试（极简框架 + 可选 GoogleTest）
#   3) Tokenizer（ASCII / JSON 词表 / BPE）
#   4) KV Cache 与 Context（Batch / Forward / RadixTree / RequestContext）
#
# 用法：
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . --config Release
#   ctest -C Release --output-on-failure
# =============================================================================

cmake_minimum_required(VERSION 3.17)
project(mini_ai CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC 默认按系统代码页解析源文件；加上 /utf-8 后中文注释不会触发 C4819
if(MSVC)
  add_compile_options(/utf-8)
endif()

# 头文件搜索路径：让 #include "mini_llama/tensor.h" 能找到 include/ 下的文件
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# ---------------------------------------------------------------------------
# 核心库源文件（Tensor + 算子 + Tokenizer + KV Cache / Context）
# ---------------------------------------------------------------------------
set(MINI_AI_CORE_SOURCES
    src/tensor.cc
    src/ops.cc
    src/matmul_optimized.cc
    src/tokenizer.cc
    src/json_vocab_tokenizer.cc
    src/bpe_tokenizer.cc
    src/gguf_tokenizer.cc
    src/model.cc
    src/loader.cc
    src/kv_cache.cc
    src/context.cc
    src/batch.cc
    src/forward.cc
    src/radix_tree.cc
    src/request_context.cc
    src/chat.cc
)

# ---------------------------------------------------------------------------
# 极简测试框架可执行文件（不依赖 GoogleTest，默认一定能编过）
# ---------------------------------------------------------------------------
set(MINI_TEST_SOURCES
    ${MINI_AI_CORE_SOURCES}
    tests/test_main.cc
    tests/test_tensor.cc
    tests/test_ops.cc
    tests/test_tokenizer.cc
    tests/test_kv_cache.cc
    tests/test_batch.cc
    tests/test_forward.cc
    tests/test_radix_tree.cc
    tests/test_request_context.cc
    tests/test_chat.cc
)

add_executable(mini-ai-tests ${MINI_TEST_SOURCES})
set_target_properties(mini-ai-tests PROPERTIES OUTPUT_NAME mini-ai-tests)

enable_testing()
add_test(NAME mini-ai-tests COMMAND mini-ai-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# ---------------------------------------------------------------------------
# GoogleTest（可选）：找不到依赖时跳过，不影响极简框架测试
# 验收标准要求「能用 GoogleTest 和极简框架写测试」，因此这里两种都保留。
# ---------------------------------------------------------------------------
option(MINI_AI_FETCH_GTEST "Download GoogleTest via FetchContent" ON)

if(MINI_AI_FETCH_GTEST)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    # 与主项目共用运行时库（Windows MSVC 需要）
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(mini-ai-gtests
        ${MINI_AI_CORE_SOURCES}
        tests/gtest/test_tensor.cc
        tests/gtest/test_ops.cc
        tests/gtest/test_tokenizer.cc
        tests/gtest/test_forward.cc
    )
    target_link_libraries(mini-ai-gtests PRIVATE GTest::gtest_main)
    set_target_properties(mini-ai-gtests PROPERTIES OUTPUT_NAME mini-ai-gtests)
    add_test(NAME mini-ai-gtests COMMAND mini-ai-gtests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
