#include <gtest/gtest.h>
#include "mini_llama/ops.h"
#include "mini_llama/tensor.h"

using mini_llama::ArgMax;
using mini_llama::Linear;
using mini_llama::Matmul;
using mini_llama::RmsNorm;
using mini_llama::Softmax;
using mini_llama::Tensor;

//[[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
TEST(OpsTest, MatmulSimple) {
  Tensor a({2, 2}, 0.0f);
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  a[3] = 4;
  Tensor b({2, 2}, 0.0f);
  b[0] = 5;
  b[1] = 6;
  b[2] = 7;
  b[3] = 8;
  Tensor c = Matmul(a, b);
  EXPECT_NEAR(c[0], 19.0f, 1e-5f);
  EXPECT_NEAR(c[1], 22.0f, 1e-5f);
  EXPECT_NEAR(c[2], 43.0f, 1e-5f);
  EXPECT_NEAR(c[3], 50.0f, 1e-5f);
}

//中间维对不上，矩阵乘要报错
TEST(OpsTest, MatmulShapeMismatch) {
  Tensor a({2, 3}, 0.0f);
  Tensor b({2, 3}, 0.0f);
  EXPECT_THROW(Matmul(a, b), std::runtime_error);
}

//单位矩阵乘进去，输出还是原来的x
TEST(OpsTest, LinearIdentity) {
  Tensor x({3}, 0.0f);
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  Tensor weight({3, 3}, 0.0f);
  weight[0] = 1;
  weight[4] = 1;
  weight[8] = 1;
  Tensor y = Linear(x, weight);
  EXPECT_NEAR(y[0], 1.0f, 1e-5f);
  EXPECT_NEAR(y[1], 2.0f, 1e-5f);
  EXPECT_NEAR(y[2], 3.0f, 1e-5f);
}

//x和weight都是1，输出还是1
TEST(OpsTest, RmsNormUnit) {
  Tensor x({4}, 1.0f);
  Tensor w({4}, 1.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(y[i], 1.0f, 1e-4f);
  }
}

//weight是2，输出就变成2
TEST(OpsTest, RmsNormWithWeight) {
  Tensor x({4}, 1.0f);
  Tensor w({4}, 2.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(y[i], 2.0f, 1e-4f);
  }
}

//[1,2,3]做softmax，加起来应该是1
TEST(OpsTest, Softmax) {
  Tensor x({3}, 0.0f);
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  Tensor y = Softmax(x);
  EXPECT_NEAR(y[0], 0.09003057f, 1e-5f);
  EXPECT_NEAR(y[1], 0.24472847f, 1e-5f);
  EXPECT_NEAR(y[2], 0.66524096f, 1e-5f);
  EXPECT_NEAR(y[0] + y[1] + y[2], 1.0f, 1e-5f);
}

//三个一样的数，每个都是1/3
TEST(OpsTest, SoftmaxUniform) {
  Tensor x({3}, 0.0f);
  Tensor y = Softmax(x);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(y[i], 1.0f / 3.0f, 1e-5f);
  }
}

//二维张量不能直接做softmax
TEST(OpsTest, SoftmaxWrongDim) {
  Tensor x({2, 3}, 0.0f);
  EXPECT_THROW(Softmax(x), std::runtime_error);
}

//两个5里取第一次出现的下标
TEST(OpsTest, ArgMax) {
  Tensor x({5}, 0.0f);
  x[0] = 1;
  x[1] = 5;
  x[2] = 3;
  x[3] = 5;
  x[4] = 2;
  EXPECT_EQ(ArgMax(x), 1);
}
