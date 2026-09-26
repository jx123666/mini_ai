# mini_ai — 核心推理模块学习实现

对照课程任务「核心推理模块」，在本仓库逐步实现：

1. **单元测试**（极简框架 + GoogleTest）
2. **Tensor 与基础算子**（matmul / rmsnorm / softmax）
3. **Tokenizer**（ASCII、JSON 词表、BPE 合并、encode/decode roundtrip）
4. **KV Cache 与 Context**（Batch / Prefill-Decode 状态 / RadixTree）

命名空间与路径刻意与 `mini-llama` 课程文档保持一致（`mini_llama/...`），方便对照阅读完整项目。

---

## 快速开始

```powershell
cd D:\SoftWareWork\ProjectWork\ProjectAI\mini_ai
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
ctest -C Release --output-on-failure
```

也可以直接跑：

```powershell
.\Release\mini-ai-tests.exe          # 极简框架
.\Release\mini-ai-gtests.exe         # GoogleTest（需联网下载依赖）
```

若不想下载 GoogleTest：

```powershell
cmake .. -DMINI_AI_FETCH_GTEST=OFF
```

---

## 主要目录

| 路径 | 作用 |
|------|------|
| `include/mini_llama/` | 头文件：Tensor、算子、Tokenizer、KV Cache、Context 等 |
| `src/` | 对应实现 |
| `tests/` | 极简框架单测 |
| `tests/gtest/` | GoogleTest 版本 |

---

## 建议阅读顺序

1. `tensor.h` → `tensor.cc`（先搞懂内存布局）
2. `matmul_optimized.cc` 里的 `MatmulNaive`
3. `ops.cc` 里的 `RmsNorm`、`Softmax`
4. `tokenizer` 相关实现与测试
5. `kv_cache` / `context` / `batch` / `forward`（Prefill 与 Decode 状态变化）
