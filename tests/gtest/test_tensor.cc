#include <gtest/gtest.h>
#include "mini_llama/tensor.h"

using mini_llama::MakeTensor1D;
using mini_llama::MakeTensor2D;
using mini_llama::MakeTensor3D;
using mini_llama::MakeTensor4D;
using mini_llama::Tensor;

//形状[3,4,5]，元素个数是60
TEST(TensorTest, ShapeAndSize) {
  Tensor t1({3, 4, 5}, 0.0f);
  EXPECT_EQ(t1.num_dims(), 3);
  EXPECT_EQ(t1.size(), 60);
  EXPECT_EQ(t1.shape[0], 3);
  EXPECT_EQ(t1.shape[1], 4);
  EXPECT_EQ(t1.shape[2], 5);
}

//填进去的数每个位置都还在
TEST(TensorTest, Fill) {
  Tensor t({10}, 3.14f);
  for (size_t i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(t.data[i], 3.14f, 1e-6f);
  }
}

//维度是0要抛异常
TEST(TensorTest, RejectsZeroDimension) {
  EXPECT_THROW(Tensor({2, 0}, 0.0f), std::runtime_error);
}

//维度是负数要抛异常
TEST(TensorTest, RejectsNegativeDimension) {
  EXPECT_THROW(Tensor({2, -3}, 0.0f), std::runtime_error);
}

//多维下标写进去再读出来要对上
TEST(TensorTest, Indexing) {
  Tensor t({2, 3}, 0.0f);
  t.At({0, 0}) = 1.0f;
  t.At({0, 1}) = 2.0f;
  t.At({1, 2}) = 6.0f;
  EXPECT_NEAR(t.At({0, 0}), 1.0f, 1e-6f);
  EXPECT_NEAR(t.At({0, 1}), 2.0f, 1e-6f);
  EXPECT_NEAR(t.At({1, 2}), 6.0f, 1e-6f);
}

//下标个数和维度不一致
TEST(TensorTest, IndexWrongRank) {
  Tensor t({2, 3}, 0.0f);
  EXPECT_THROW(t.At({1}), std::runtime_error);
}

//下标太大或是负数都越界
TEST(TensorTest, IndexOutOfRange) {
  Tensor t({2, 3}, 0.0f);
  EXPECT_THROW(t.At({0, 3}), std::out_of_range);
  EXPECT_THROW(t.At({-1, 0}), std::out_of_range);
}

//1维到4维的快捷构造，元素个数是各维相乘
TEST(TensorTest, MakeHelpers) {
  EXPECT_EQ(MakeTensor1D(10).shape[0], 10);
  EXPECT_EQ(MakeTensor2D(3, 4).size(), 12);
  EXPECT_EQ(MakeTensor3D(2, 3, 4).size(), 24);
  EXPECT_EQ(MakeTensor4D(2, 3, 4, 5).size(), 120);
}

//第1行第3列是13，RowPtr拿到这一行的开头是10
TEST(TensorTest, At2AndRowPtr) {
  Tensor t({3, 4}, 0.0f);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      t.At2(i, j) = static_cast<float>(i * 10 + j);
    }
  }
  EXPECT_NEAR(t.At2(1, 3), 13.0f, 1e-6f);
  EXPECT_NEAR(t.RowPtr(1)[0], 10.0f, 1e-6f);
}

//6个数可以改成[3,2]，不能改成[4,2]
TEST(TensorTest, ReshapeChecked) {
  Tensor t({2, 3}, 0.0f);
  for (int i = 0; i < 6; ++i) {
    t[i] = static_cast<float>(i);
  }
  Tensor r = t.ReshapeChecked({3, 2}, "gtest");
  EXPECT_EQ(r.shape[0], 3);
  EXPECT_EQ(r.shape[1], 2);
  EXPECT_NEAR(r.At2(2, 1), 5.0f, 1e-6f);
  EXPECT_THROW(t.ReshapeChecked({4, 2}, "gtest"), std::runtime_error);
}
