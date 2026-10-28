# mini-ai 的构建与测试入口。

cmake_minimum_required(VERSION 3.17)
project(mini_ai CXX)
find_package(Threads REQUIRED)

option(MINI_AI_CUDA "Enable the NVIDIA CUDA backend" OFF)
if(MINI_AI_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_compile_definitions(MINI_AI_CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC 默认按系统代码页解析源文件；加上 /utf-8 后中文注释不会触发 C4819
if(MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>" "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()

# 头文件搜索路径：让 #include "mini_llama/tensor.h" 能找到 include/ 下的文件
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# 核心库源文件（Tensor + 算子 + Tokenizer + KV Cache / Context + Chat）
set(MINI_AI_CORE_SOURCES
    src/tensor.cc
    src/ops.cc
    src/matmul_optimized.cc
    src/thread_pool.cc
    src/quantized_tensor.cc
    src/runtime.cc
    src/debug.cc
    src/backend.cc
    src/tokenizer.cc
    src/json_vocab_tokenizer.cc
    src/bpe_tokenizer.cc
    src/gguf_tokenizer.cc
    src/model.cc
    src/loader.cc
    src/gguf.cc
    src/gguf_loader.cc
    src/kv_cache.cc
    src/context.cc
    src/batch.cc
    src/forward.cc
    src/radix_tree.cc
    src/request_context.cc
    src/chat.cc
    src/prompt_builder.cc
    src/terminal.cc
    src/sampler.cc
)

if(MINI_AI_CUDA)
  list(APPEND MINI_AI_CORE_SOURCES src/cuda_backend.cu)
endif()

# 极简测试框架可执行文件
set(MINI_TEST_SOURCES
    ${MINI_AI_CORE_SOURCES}
    tests/test_main.cc
    tests/test_compute.cc
    tests/test_cuda.cc
    tests/test_prompt_template.cc
    tests/test_tensor.cc
    tests/test_ops.cc
    tests/test_tokenizer.cc
    tests/test_kv_cache.cc
    tests/test_batch.cc
    tests/test_forward.cc
    tests/test_radix_tree.cc
    tests/test_request_context.cc
    tests/test_chat.cc
    tests/test_sampler.cc
    tests/test_loader.cc
    tests/test_gguf.cc
    tests/test_gguf_loader.cc
    tests/test_gguf_tokenizer.cc
)

add_executable(mini-ai-tests ${MINI_TEST_SOURCES})
set_target_properties(mini-ai-tests PROPERTIES OUTPUT_NAME mini-ai-tests)

add_executable(mini-ai ${MINI_AI_CORE_SOURCES} src/main.cc)
set_target_properties(mini-ai PROPERTIES OUTPUT_NAME mini-ai)
target_link_libraries(mini-ai PRIVATE Threads::Threads)
target_link_libraries(mini-ai-tests PRIVATE Threads::Threads)
if(MINI_AI_CUDA)
  target_link_libraries(mini-ai PRIVATE CUDA::cudart)
  target_link_libraries(mini-ai-tests PRIVATE CUDA::cudart)
endif()

enable_testing()
add_test(NAME mini-ai-tests COMMAND mini-ai-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
                  RESULT_VARIABLE MINI_AI_NUMPY_STATUS OUTPUT_QUIET ERROR_QUIET)
  if(MINI_AI_NUMPY_STATUS EQUAL 0)
    add_test(NAME golden-logits COMMAND "${Python3_EXECUTABLE}"
             "${CMAKE_SOURCE_DIR}/scripts/test_golden.py" --exe $<TARGET_FILE:mini-ai>)
    add_test(NAME cli-regression COMMAND "${Python3_EXECUTABLE}"
             "${CMAKE_SOURCE_DIR}/scripts/test_cli.py" --exe $<TARGET_FILE:mini-ai>)
    set_tests_properties(golden-logits cli-regression PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  endif()
endif()

# GoogleTest：找不到依赖时跳过，不影响极简框架测试
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
    target_link_libraries(mini-ai-gtests PRIVATE GTest::gtest_main Threads::Threads)
    if(MINI_AI_CUDA)
      target_link_libraries(mini-ai-gtests PRIVATE CUDA::cudart)
    endif()
    set_target_properties(mini-ai-gtests PROPERTIES OUTPUT_NAME mini-ai-gtests)
    add_test(NAME mini-ai-gtests COMMAND mini-ai-gtests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
