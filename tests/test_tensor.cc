//测Tensor：形状、填充、下标、越界、reshape。和gtest版测的是同一批事
#include "mini_llama/tensor.h"
#include "tests/test_main.h"

using mini_llama::MakeTensor1D;
using mini_llama::MakeTensor2D;
using mini_llama::MakeTensor3D;
using mini_llama::MakeTensor4D;
using mini_llama::Tensor;

//形状[3,4,5]，元素个数应该是3*4*5=60
static bool TestTensorShapeAndSize() {
  Tensor t1({3, 4, 5}, 0.0f);
  MINI_LLAMA_ASSERT_EQ(t1.num_dims(), 3);
  MINI_LLAMA_ASSERT_EQ(t1.size(), 60);
  MINI_LLAMA_ASSERT_EQ(t1.shape[0], 3);
  MINI_LLAMA_ASSERT_EQ(t1.shape[1], 4);
  MINI_LLAMA_ASSERT_EQ(t1.shape[2], 5);
  return true;
}

//构造时填的数，每个元素都应该是这个值
static bool TestTensorFill() {
  Tensor t({10}, 3.14f);
  for (size_t i = 0; i < t.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(t.data[i], 3.14f, 1e-6f);
  }
  return true;
}

//某一维是0，应该直接报错
static bool TestTensorRejectsZeroDimension() {
  try {
    Tensor t({2, 0}, 0.0f);
    MINI_LLAMA_ASSERT_FAIL("expected exception for zero tensor dimension");
  } catch (const std::runtime_error&) {
  }
  return true;
}

//某一维是负数，也应该报错
static bool TestTensorRejectsNegativeDimension() {
  try {
    Tensor t({2, -3}, 0.0f);
    MINI_LLAMA_ASSERT_FAIL("expected exception for negative tensor dimension");
  } catch (const std::runtime_error&) {
  }
  return true;
}

//[2,3]里At({1,2})的一维下标是1*3+2=5
static bool TestTensorIndexing() {
  Tensor t({2, 3}, 0.0f);
  t.At({0, 0}) = 1.0f;
  t.At({0, 1}) = 2.0f;
  t.At({1, 2}) = 6.0f;
  MINI_LLAMA_ASSERT_NEAR(t.At({0, 0}), 1.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(t.At({0, 1}), 2.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(t.At({1, 2}), 6.0f, 1e-6f);
  return true;
}

//二维张量只给一个下标，维度对不上
static bool TestTensorIndexWrongRank() {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At({1});
    MINI_LLAMA_ASSERT_FAIL("expected exception for wrong index rank");
  } catch (const std::runtime_error&) {
  }
  return true;
}

//下标不能是负数
static bool TestTensorIndexNegative() {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At({-1, 0});
    MINI_LLAMA_ASSERT_FAIL("expected exception for negative index");
  } catch (const std::out_of_range&) {
  }
  return true;
}

//列只有0、1、2，下标3越界
static bool TestTensorIndexTooLarge() {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At({0, 3});
    MINI_LLAMA_ASSERT_FAIL("expected exception for index above bound");
  } catch (const std::out_of_range&) {
  }
  return true;
}

//一维张量用[]直接按下标读写
static bool TestTensor1dAccess() {
  Tensor t({5}, 0.0f);
  t[2] = 42.0f;
  MINI_LLAMA_ASSERT_NEAR(t[2], 42.0f, 1e-6f);
  return true;
}

//MakeTensor1D到4D只是少写一层花括号，元素个数仍是各维相乘
static bool TestMakeTensorHelpers() {
  auto t1 = MakeTensor1D(10, 1.0f);
  MINI_LLAMA_ASSERT_EQ(t1.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(t1.shape[0], 10);

  auto t2 = MakeTensor2D(3, 4, 1.0f);
  MINI_LLAMA_ASSERT_EQ(t2.num_dims(), 2);
  MINI_LLAMA_ASSERT_EQ(t2.size(), 12);

  auto t3 = MakeTensor3D(2, 3, 4, 1.0f);
  MINI_LLAMA_ASSERT_EQ(t3.size(), 24);

  auto t4 = MakeTensor4D(2, 3, 4, 5, 1.0f);
  MINI_LLAMA_ASSERT_EQ(t4.size(), 120);
  return true;
}

//At2(行,列)和At({行,列})是同一格
static bool TestTensorAt2() {
  Tensor t({2, 3}, 0.0f);
  t.At2(0, 1) = 5.0f;
  t.At2(1, 2) = 9.0f;
  MINI_LLAMA_ASSERT_NEAR(t.At2(0, 1), 5.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(t.At2(1, 2), 9.0f, 1e-6f);
  return true;
}

//第1行开头应该是10、11、12、13，RowPtr直接拿到这一行
static bool TestTensorRowPtr() {
  Tensor t({3, 4}, 0.0f);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      t.At2(i, j) = static_cast<float>(i * 10 + j);
    }
  }
  const float* row1 = t.RowPtr(1);
  MINI_LLAMA_ASSERT_NEAR(row1[0], 10.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(row1[3], 13.0f, 1e-6f);
  return true;
}

//形状和预期一样，不抛异常
static bool TestTensorAssertShapePass() {
  Tensor t({2, 3, 4}, 0.0f);
  t.AssertShape({2, 3, 4}, "TestPass");
  return true;
}

//最后一维4和5对不上，应该抛异常
static bool TestTensorAssertShapeFail() {
  Tensor t({2, 3, 4}, 0.0f);
  try {
    t.AssertShape({2, 3, 5}, "TestFail");
    MINI_LLAMA_ASSERT_FAIL("expected exception for shape mismatch");
  } catch (const std::runtime_error&) {
  }
  return true;
}

//[2,3]和[3,2]都是6个数，reshape只改看法，不改数据顺序
static bool TestTensorReshapeCheckedPass() {
  Tensor t({2, 3}, 0.0f);
  for (int i = 0; i < 6; ++i) {
    t[i] = static_cast<float>(i);
  }
  Tensor r = t.ReshapeChecked({3, 2}, "TestReshape");
  MINI_LLAMA_ASSERT_EQ(r.shape[0], 3);
  MINI_LLAMA_ASSERT_EQ(r.shape[1], 2);
  MINI_LLAMA_ASSERT_NEAR(r.At2(0, 0), 0.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(r.At2(2, 1), 5.0f, 1e-6f);
  return true;
}

//6个数变不成4*2=8个数
static bool TestTensorReshapeCheckedFail() {
  Tensor t({2, 3}, 0.0f);
  try {
    t.ReshapeChecked({4, 2}, "TestReshapeFail");
    MINI_LLAMA_ASSERT_FAIL("expected exception for incompatible reshape");
  } catch (const std::runtime_error&) {
  }
  return true;
}

//程序启动前这个对象就会构造，顺手把上面的测试都登记上
static struct TensorTestRegistrar {
  TensorTestRegistrar() {
    RegisterTest("tensor_shape_and_size", TestTensorShapeAndSize);
    RegisterTest("tensor_fill", TestTensorFill);
    RegisterTest("tensor_rejects_zero_dimension",
                 TestTensorRejectsZeroDimension);
    RegisterTest("tensor_rejects_negative_dimension",
                 TestTensorRejectsNegativeDimension);
    RegisterTest("tensor_indexing", TestTensorIndexing);
    RegisterTest("tensor_index_wrong_rank", TestTensorIndexWrongRank);
    RegisterTest("tensor_index_negative", TestTensorIndexNegative);
    RegisterTest("tensor_index_too_large", TestTensorIndexTooLarge);
    RegisterTest("tensor_1d_access", TestTensor1dAccess);
    RegisterTest("make_tensor_helpers", TestMakeTensorHelpers);
    RegisterTest("tensor_at2", TestTensorAt2);
    RegisterTest("tensor_row_ptr", TestTensorRowPtr);
    RegisterTest("tensor_assert_shape_pass", TestTensorAssertShapePass);
    RegisterTest("tensor_assert_shape_fail", TestTensorAssertShapeFail);
    RegisterTest("tensor_reshape_checked_pass", TestTensorReshapeCheckedPass);
    RegisterTest("tensor_reshape_checked_fail", TestTensorReshapeCheckedFail);
  }
} g_tensor_test_registrar;
