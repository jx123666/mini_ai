// =============================================================================
// 文件作用：tensor.h
//   定义推理引擎最底层的数据结构 —— 稠密浮点张量 Tensor。
//   在大模型推理里，激活值、权重、注意力分数等几乎都以 Tensor 形式流动。
//
// 设计要点（对应验收：理解张量结构）：
//   1. 逻辑上是多维数组（由 shape 描述）
//   2. 物理上是一维连续内存（data: vector<float>），采用 row-major 行优先布局
//   3. 提供 FlatIndex / At / At2 等访问器，把「多维下标」映射成「一维偏移」
//
// 与 mini-llama 课程文档对应：
//   include/mini_llama/tensor.h  /  src/tensor.cc
// =============================================================================

#ifndef INCLUDE_MINI_LLAMA_TENSOR_H_
#define INCLUDE_MINI_LLAMA_TENSOR_H_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {

// ---------------------------------------------------------------------------
// Tensor：最小可用的 F32 稠密张量
//
// 例子：
//   Tensor t({2, 3}, 0.0f);   // 2 行 3 列，全部填 0
//   t.At2(1, 2) = 5.0f;      // 第 1 行第 2 列赋值为 5
//   // 内存布局: [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2]
// ---------------------------------------------------------------------------
struct Tensor {
  std::vector<float> data;  // 连续存放的全部元素
  std::vector<int> shape;   // 各维长度，如 {m, k} 表示矩阵

  Tensor() = default;

  // 按 shape 分配内存，并用 fill 填充
  explicit Tensor(const std::vector<int>& input_shape, float fill = 0.0f);

  // 元素总数 = shape 各维之积
  size_t size() const { return data.size(); }
  size_t num_elements() const { return data.size(); }

  // 维度个数，例如矩阵为 2
  int num_dims() const { return static_cast<int>(shape.size()); }

  // 多维下标 → 一维 flat index（行优先）
  size_t FlatIndex(const std::vector<int>& indices) const;

  // 通用多维访问
  float& At(const std::vector<int>& indices);
  float At(const std::vector<int>& indices) const;

  // 固定维度的便捷访问（维度不对会抛异常，便于调试）
  float& At1(int i);
  float At1(int i) const;
  float& At2(int i, int j);
  float At2(int i, int j) const;
  float& At3(int i, int j, int k);
  float At3(int i, int j, int k) const;
  float& At4(int i, int j, int k, int l);
  float At4(int i, int j, int k, int l) const;

  // 2D 张量第 row 行的起始指针，常用于 matmul / SIMD 点积
  float* RowPtr(int row);
  const float* RowPtr(int row) const;

  // 形状断言：与预期不符时抛异常，帮助定位推理中的 shape bug
  void AssertShape(const std::vector<int>& expected, const char* caller) const;

  // 改 shape，但元素个数必须不变（只改「视图」，不改数据）
  Tensor ReshapeChecked(const std::vector<int>& new_shape,
                        const char* caller) const;

  // 一维下标直达 data（最快，但调用者需自己保证不越界）
  float& operator[](size_t i) { return data[i]; }
  float operator[](size_t i) const { return data[i]; }

  std::string ShapeStringShort() const;
  std::string ShapeString() const { return ShapeStringShort(); }
  void Print(const std::string& name = "", bool print_data = false) const;
};

// 便捷工厂：直接按维度创建
Tensor MakeTensor1D(int d0, float fill = 0.0f);
Tensor MakeTensor2D(int d0, int d1, float fill = 0.0f);
Tensor MakeTensor3D(int d0, int d1, int d2, float fill = 0.0f);
Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill = 0.0f);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_TENSOR_H_
