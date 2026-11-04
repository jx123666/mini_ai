# mini_ai — C++ 大语言模型推理项目

使用 C++17 实现张量、基础算子、分词和推理状态管理，并逐步扩展模型加载、文本生成、量化计算及数值验证。

## 分支说明

- main 保留早期核心模块实现，包含 Tensor、基础算子、Tokenizer、KV Cache、Context、Batch 和测试。
- [submit-core](https://github.com/jx123666/mini_ai/tree/submit-core) 包含较完整的 CPU 推理链路、GGUF 与 JSON+BIN 模型加载、终端聊天、权重量化、线程池与 SSE2 优化。

查看简历中描述的 GGUF 加载、量化及文本生成实现，请进入 submit-core 分支。该分支的 README 提供详细构建、运行与验证方法。

## main 分支的主要模块

- 张量与基础算子，包括矩阵乘法、RMSNorm 和 Softmax。
- 分词，包括 ASCII、JSON 词表及 BPE 合并。
- KV Cache 与上下文管理，包括 Batch 和 Prefill/Decode 状态。
- 基础测试与可选 GoogleTest。

## 构建 main 分支

    cmake -S . -B build -DMINI_AI_FETCH_GTEST=OFF
    cmake --build build --config Release
    ctest --test-dir build -C Release --output-on-failure

## 验证范围

性能与 NumPy 数值对齐使用 tiny 测试模型。较完整分支包含 LLaMA、Qwen2 架构的 GGUF 映射逻辑，具体预训练模型的兼容性需要逐模型验证。tiny 测试结果应与实际大模型的吞吐、内存占用和生成质量分别评估。
