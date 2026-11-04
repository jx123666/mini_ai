# mini_ai — C++ 大语言模型推理项目

C++17 LLaMA-style 推理内核，支持 CPU 文本生成、终端多轮对话和 GGUF 模型加载。

## 构建

需要 CMake 3.17 及以上版本和支持 C++17 的编译器。

```powershell
cmake -S . -B build/cpu -DMINI_AI_FETCH_GTEST=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu --config Release
ctest --test-dir build/cpu -C Release --output-on-failure
```

Visual Studio 的可执行文件位于 `build/cpu/Release/`；单配置生成器位于 `build/cpu/`。
`MINI_AI_FETCH_GTEST=ON` 会额外下载并运行 GoogleTest。
Python 环境装有 NumPy 时，CTest 会注册 logits 数值对齐和命令行回归测试。
可以通过 `-DPython3_EXECUTABLE=<python路径>` 指定解释器。

## 运行

以下命令以单配置生成器的输出路径为例：

```powershell
./build/cpu/mini-ai.exe generate --model models/tiny -p hello -n 8
./build/cpu/mini-ai.exe run models/chat/model.gguf -n 32 --temperature 0.7
./build/cpu/mini-ai.exe bench models/tiny -p hello -n 8 --threads 4 --verbose
./build/cpu/mini-ai.exe generate --model models/tiny --quant q8_0 -n 8
./build/cpu/mini-ai.exe inspect models/tiny/model.json
./build/cpu/mini-ai.exe inspect-gguf models/chat/model.gguf
./build/cpu/mini-ai.exe tokenize-gguf models/chat/model.gguf "你好，世界"
```

`generate`、`run`、`bench` 共用模型加载入口。模型来源可以是 GGUF 文件、JSON+BIN 目录或权重文件；
直接指定 BIN 时可通过 `--config` 指定清单。目录同时包含 JSON+BIN 和 GGUF 时使用 JSON+BIN；
只包含 GGUF 时要求目录中恰好有一份 GGUF，避免选错模型。

生成支持 `--temperature`、`--top-k`、`--seed`。
CPU 可通过 `--threads` 设置线程数，0 自动选择，1 关闭并行。
`generate` 和 `bench` 还支持 `--matmul naive|threaded|simd|threaded-simd`，默认 `threaded-simd`。
SIMD 使用 SSE2；其他平台回退到标量点积。

聊天命令包括 `/help`、`/clear`、`/stats`、`/params` 和 `/exit`。
`--template plain|qwen2|<gguf路径>` 可覆盖默认聊天模板。

## 模型与分词

- Transformer：GQA、RMSNorm、SwiGLU、KV Cache、普通及 NeoX RoPE、Q/K/V bias。
- GGUF：LLaMA/Qwen2 张量映射，支持 F32、F16、Q8_0、Q4_0、Q4_1。
- 量化线性层保留压缩块，逐块解码累加；embedding 和归一化参数使用 F32。
- `--quant q8_0|q4_0|q4_1` 可以在运行前量化线性层权重。
- Tokenizer：ASCII、JSON 最长匹配词表、GPT-2 字节 BPE、Qwen2/GPT-2 Unicode 预分词。
- GGUF 分词器读取词表、merges、特殊 token 和 BOS/EOS 开关；不支持的分词器会报错。
- 模板支持消息循环、条件、字符串拼接、转义和空白控制，可执行常见 Qwen2 文本聊天模板。

模板解析器不是完整 Jinja 引擎，暂不支持工具调用、过滤器、`set`、`else` 和嵌套循环。
SentencePiece、其他架构、局部或缩放 RoPE、滑动窗口、K-quants 和 IQ-quants 不在当前支持范围内。
仓库内只有 tiny 模型和测试数据；大型模型需要另行准备。

## 数值对齐与性能统计

```powershell
./build/cpu/mini-ai.exe generate --model models/tiny -p hello -n 8 --dump-logits build/cpp-logits
python scripts/reference_forward.py --tokens 1,104,101,108,108,111 -n 8 --output-dir build/py-logits
python scripts/compare_logits.py --cpp-dir build/cpp-logits --py-dir build/py-logits
```

每一步输出 `logits_0000.bin` 等 F32 文件，以及记录输入和生成序列的 `tokens.json`。
复用输出目录前请清理旧文件，避免混入上一次运行留下的步骤。
参考脚本通过 `--tokens` 接受明确的 token 序列，避免分词差异影响数值比较。

`bench` 输出模型加载、分词、prefill、decode、采样耗时和 token/s。
decode 吞吐量按实际执行的 decode forward 次数计算，不包含采样和终端输出耗时。
`--verbose` 输出当前位置和 logits top-5。

## CUDA

CUDA 为可选构建，默认 CPU 构建无需 NVIDIA 依赖。
CUDA 构建需要完整 CUDA Toolkit 和受 NVIDIA 支持的宿主编译器。

```powershell
cmake -S . -B build/cuda -DMINI_AI_CUDA=ON -DMINI_AI_FETCH_GTEST=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda --config Release
./build/cuda/mini-ai.exe --cuda-info
./build/cuda/mini-ai.exe bench models/tiny --backend cuda --device 0 -n 8
ctest --test-dir build/cuda -C Release --output-on-failure
python scripts/run_cuda_benchmarks.py --exe build/cuda/mini-ai.exe --model models/tiny
```

CUDA 路径包含 F32/量化线性层、RMSNorm、RoPE、SwiGLU、残差、GPU KV Cache 和 attention。
权重、激活和 KV Cache 保留在设备上，每步将最终 logits 回传 CPU 采样。
统计输出包括上传权重字节数、linear/activation/attention 调用数和 logits 回传字节数。
实现使用自有 CUDA kernel，当前未接入 cuBLAS。

CUDA 源码尚未在完整 CUDA 工具链或真实 GPU 上验证。CUDA 构建中的对照测试会比较两种 RoPE 布局、
F32 及三种量化格式的多步 CPU/GPU logits；设备初始化失败会使测试失败，不会作为通过跳过。

## 主要文件

| 路径 | 内容 |
|---|---|
| `src/forward.cc` | CPU Transformer 推理 |
| `src/runtime.cc` | 模型来源和分词器加载 |
| `src/gguf_loader.cc` | GGUF 配置、权重映射和格式校验 |
| `src/quantized_tensor.cc` | 压缩权重、量化和点积 |
| `src/matmul_optimized.cc` | 标量、线程及 SIMD 调度 |
| `src/thread_pool.cc` | 常驻工作线程 |
| `src/cuda_backend.cu` | CUDA 权重、状态和推理算子 |
| `scripts/test_golden.py` | CPU/NumPy 多步数值回归 |
| `scripts/test_cli.py` | GGUF、量化、聊天及参数回归 |
