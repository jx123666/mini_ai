#include "mini_llama/tensor.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
namespace mini_llama {
namespace {
const char* CallerName(const char* caller) {
  return caller == nullptr ? "Tensor" : caller;
}

//把shape格式化成[2, 3, 4]，异常信息里会用到
std::string ShapeToString(const std::vector<int>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      s += ", ";
    }
    s += std::to_string(shape[i]);
  }
  s += "]";
  return s;
}

//计算元素总数，并检查：
//每一维必须 > 0
//乘法不能溢出size_t
size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
  size_t total = 1;
  //遍历shape的每个维度
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    const int dim = shape[axis];
    if (dim <= 0) {//如果维度小于等于0，异常
      throw std::runtime_error(std::string(CallerName(caller)) +
                               ": dimension at axis " + std::to_string(axis) +
                               " must be positive, got " + std::to_string(dim) +
                               " in shape " + ShapeToString(shape));
    }
    //将维度转换为size_t
    const size_t dim_size = static_cast<size_t>(dim);
    if (total > std::numeric_limits<size_t>::max() / dim_size) {
      throw std::runtime_error(std::string(CallerName(caller)) +
                               ": shape element count overflow for " +
                               ShapeToString(shape));
    }
    total *= dim_size;
  }
  return total;
}


void CheckRank(const Tensor& t, int expected, const char* caller) {
  if (t.quantized) throw std::runtime_error("dequantize tensor before accessing F32 elements");
  //如果tensor的维度不等于预期的维度，异常
  if (t.num_dims() != expected) {
    throw std::runtime_error(std::string(caller) + " expected " +
                             std::to_string(expected) + "D tensor, got " +
                             t.ShapeStringShort());
  }
}
//检查轴索引是否在范围内
//输入tensor的维度为t.shape[axis]
//输入轴索引为flat_index
//输入调用者为caller
void CheckAxisIndex(const Tensor& t, int axis, int flat_index,
                    const char* caller) {
  const int dim = t.shape[axis];//获取tensor的维度
  if (flat_index < 0 || flat_index >= dim) {//如果轴索引不在范围内，异常
    throw std::out_of_range(
        std::string(caller) + ": index " + std::to_string(flat_index) +
        " out of range for axis " + std::to_string(axis) + " with size " +
        std::to_string(dim) + " in tensor " + t.ShapeStringShort());
  }
}

} 

Tensor::Tensor(const std::vector<int>& input_shape, float fill)
    : shape(input_shape) {
  const size_t total = CheckedNumel(shape, "Tensor constructor");
  data.resize(total, fill);
}
//FlatIndex实现
//输入indices的形状为indices.size()
//输入调用者为caller
size_t Tensor::FlatIndex(const std::vector<int>& indices) const {
  if (quantized) throw std::runtime_error("dequantize tensor before accessing F32 elements");
  if (indices.size() != shape.size()) {
    throw std::runtime_error("Tensor::FlatIndex expected " +
                             std::to_string(shape.size()) +
                             " indices for tensor " + ShapeStringShort() +
                             ", got " + std::to_string(indices.size()));
  }
  for (size_t axis = 0; axis < indices.size(); ++axis) {
    CheckAxisIndex(*this, static_cast<int>(axis), indices[axis],
                   "Tensor::FlatIndex");
  }

  size_t flat = 0;
  size_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {//从最后一维往前累乘stride
    flat += static_cast<size_t>(indices[i]) * stride;
    stride *= static_cast<size_t>(shape[i]);
  }
  return flat;
}
//At实现
float& Tensor::At(const std::vector<int>& indices) {
  return data[FlatIndex(indices)];//返回FlatIndex的值
}
float Tensor::At(const std::vector<int>& indices) const {
  return data[FlatIndex(indices)];
}
//At1实现
float& Tensor::At1(int i) {
  CheckRank(*this, 1, "At1");
  CheckAxisIndex(*this, 0, i, "At1");
  return data[i];
}

float Tensor::At1(int i) const {
  CheckRank(*this, 1, "At1");
  CheckAxisIndex(*this, 0, i, "At1");
  return data[i];
}

//2D:flat=i*cols+j
float& Tensor::At2(int i, int j) {
  CheckRank(*this, 2, "At2");
  CheckAxisIndex(*this, 0, i, "At2");
  CheckAxisIndex(*this, 1, j, "At2");
  return data[i * shape[1] + j];
}

float Tensor::At2(int i, int j) const {
  CheckRank(*this, 2, "At2");
  CheckAxisIndex(*this, 0, i, "At2");
  CheckAxisIndex(*this, 1, j, "At2");
  return data[i * shape[1] + j];
}

float& Tensor::At3(int i, int j, int k) {
  CheckRank(*this, 3, "At3");
  CheckAxisIndex(*this, 0, i, "At3");
  CheckAxisIndex(*this, 1, j, "At3");
  CheckAxisIndex(*this, 2, k, "At3");
  return data[(i * shape[1] + j) * shape[2] + k];
}

float Tensor::At3(int i, int j, int k) const {
  CheckRank(*this, 3, "At3");
  CheckAxisIndex(*this, 0, i, "At3");
  CheckAxisIndex(*this, 1, j, "At3");
  CheckAxisIndex(*this, 2, k, "At3");
  return data[(i * shape[1] + j) * shape[2] + k];
}

float& Tensor::At4(int i, int j, int k, int l) {
  CheckRank(*this, 4, "At4");
  CheckAxisIndex(*this, 0, i, "At4");
  CheckAxisIndex(*this, 1, j, "At4");
  CheckAxisIndex(*this, 2, k, "At4");
  CheckAxisIndex(*this, 3, l, "At4");
  return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}

float Tensor::At4(int i, int j, int k, int l) const {
  CheckRank(*this, 4, "At4");
  CheckAxisIndex(*this, 0, i, "At4");
  CheckAxisIndex(*this, 1, j, "At4");
  CheckAxisIndex(*this, 2, k, "At4");
  CheckAxisIndex(*this, 3, l, "At4");
  return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}
//RowPtr实现
float* Tensor::RowPtr(int row) {
  CheckRank(*this, 2, "RowPtr");
  CheckAxisIndex(*this, 0, row, "RowPtr");
  return data.data() + row * shape[1];
}
//RowPtr const实现，与RowPtr类似
const float* Tensor::RowPtr(int row) const {
  CheckRank(*this, 2, "RowPtr");
  CheckAxisIndex(*this, 0, row, "RowPtr");
  return data.data() + row * shape[1];
}
//AssertShape实现
void Tensor::AssertShape(const std::vector<int>& expected,
                         const char* caller) const {
  if (shape != expected) {
    throw std::runtime_error(
        std::string(CallerName(caller)) + ": shape mismatch. expected " +
        ShapeToString(expected) + ", got " + ShapeStringShort());//异常信息
  }
}
//ReshapeChecked实现
Tensor Tensor::ReshapeChecked(const std::vector<int>& new_shape,
                              const char* caller) const {
  const size_t new_total = CheckedNumel(new_shape, CallerName(caller));
  //如果新的元素总数不等于data的元素总数，异常
  if (new_total != data.size()) {
    throw std::runtime_error(
        std::string(CallerName(caller)) + ": cannot reshape tensor with " +
        std::to_string(data.size()) + " elements into shape with " +
        std::to_string(new_total) + " elements");
  }
  Tensor r = *this;//复制当前tensor
  r.shape = new_shape;//设置新的形状
  return r;
}
//ShapeStringShort实现
std::string Tensor::ShapeStringShort() const { return ShapeToString(shape); }

void Tensor::Print(const std::string& name, bool print_data) const {
  if (!name.empty()) {
    std::cout << name << " ";
  }
  std::cout << "shape=" << ShapeStringShort() << " size=" << size()
            << std::endl;
  if (print_data) {//如果需要打印数据，打印数据
    for (size_t i = 0; i < data.size(); ++i) {
      std::cout << std::fixed << std::setprecision(6) << data[i];
      if (i + 1 < data.size()) {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }
}

Tensor MakeTensor1D(int d0, float fill) { return Tensor({d0}, fill); }

Tensor MakeTensor2D(int d0, int d1, float fill) {
  return Tensor({d0, d1}, fill);
}

Tensor MakeTensor3D(int d0, int d1, int d2, float fill) {
  return Tensor({d0, d1, d2}, fill);
}

Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill) {
  return Tensor({d0, d1, d2, d3}, fill);
}

} 
