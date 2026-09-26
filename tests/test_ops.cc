//测算子：matmul、linear、rmsnorm、softmax，还有形状不对时要报错
#include "mini_llama/ops.h"
#include "mini_llama/tensor.h"
#include "tests/test_main.h"

using mini_llama::ArgMax;
using mini_llama::ElementwiseMul;
using mini_llama::Linear;
using mini_llama::Matmul;
using mini_llama::RmsNorm;
using mini_llama::Silu;
using mini_llama::Softmax;
using mini_llama::SwiGlu;
using mini_llama::Tensor;

//单位矩阵乘任何矩阵，结果还是原来的矩阵
static bool TestMatmulIdentity() {
  Tensor a({2, 2}, 0.0f);
  a[0] = 1;
  a[1] = 0;
  a[2] = 0;
  a[3] = 1;
  Tensor b({2, 2}, 0.0f);
  b[0] = 1;
  b[1] = 2;
  b[2] = 3;
  b[3] = 4;
  Tensor c = Matmul(a, b);
  MINI_LLAMA_ASSERT_NEAR(c[0], 1.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[1], 2.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[2], 3.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[3], 4.0f, 1e-5f);
  return true;
}

//[[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
static bool TestMatmulSimple() {
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
  MINI_LLAMA_ASSERT_NEAR(c[0], 19.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[1], 22.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[2], 43.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[3], 50.0f, 1e-5f);
  return true;
}

//[2,3]乘[3,4]得到[2,4]，中间那一维必须相等
static bool TestMatmulRectangular() {
  // [2,3] x [3,4] -> [2,4]
  Tensor a({2, 3}, 0.0f);
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  a[3] = 4;
  a[4] = 5;
  a[5] = 6;
  Tensor b({3, 4}, 0.0f);
  for (int i = 0; i < 12; ++i) {
    b[i] = static_cast<float>(i);
  }
  Tensor c = Matmul(a, b);
  MINI_LLAMA_ASSERT_EQ(c.shape[0], 2);
  MINI_LLAMA_ASSERT_EQ(c.shape[1], 4);
  // c[0,0] = 1*0 + 2*4 + 3*8 = 32
  MINI_LLAMA_ASSERT_NEAR(c.At2(0, 0), 32.0f, 1e-5f);
  // c[1,0] = 4*0 + 5*4 + 6*8 = 68
  MINI_LLAMA_ASSERT_NEAR(c.At2(1, 0), 68.0f, 1e-5f);
  return true;
}

//[2,3]和[2,3]中间维对不上，不能乘
static bool TestMatmulShapeMismatch() {
  Tensor a({2, 3}, 0.0f);
  Tensor b({2, 3}, 0.0f);
  try {
    Matmul(a, b);
    MINI_LLAMA_ASSERT_FAIL("expected exception for Matmul shape mismatch");
  } catch (const std::runtime_error&) {
  }
  return true;
}

//权重是单位矩阵时，Linear输出应该等于输入
static bool TestLinear1d() {
  Tensor x({3}, 0.0f);
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  Tensor weight({3, 3}, 0.0f);
  weight[0] = 1;
  weight[4] = 1;
  weight[8] = 1;  //对角线是1，其它是0，就是单位矩阵
  Tensor y = Linear(x, weight);
  MINI_LLAMA_ASSERT_NEAR(y[0], 1.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[1], 2.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[2], 3.0f, 1e-5f);
  return true;
}

//x全是1，weight也全是1，rms大约是1，输出还是1
static bool TestRmsNormUnit() {
  Tensor x({4}, 1.0f);
  Tensor w({4}, 1.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  for (int i = 0; i < 4; ++i) {
    MINI_LLAMA_ASSERT_NEAR(y[i], 1.0f, 1e-4f);
  }
  return true;
}

//x全是1，再乘weight=2，输出应该全是2
static bool TestRmsNormWithWeight() {
  Tensor x({4}, 1.0f);
  Tensor w({4}, 2.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  for (int i = 0; i < 4; ++i) {
    MINI_LLAMA_ASSERT_NEAR(y[i], 2.0f, 1e-4f);
  }
  return true;
}

//x长度4，weight长度3，对不上
static bool TestRmsNormShapeMismatch() {
  Tensor x({4}, 0.0f);
  Tensor w({3}, 0.0f);
  try {
    RmsNorm(x, w, 1e-5f);
    MINI_LLAMA_ASSERT_FAIL("expected exception for RmsNorm shape mismatch");
  } catch (const std::runtime_error&) {
  }
  return true;
}

//Softmax([1,2,3])，三个数都大于0，加起来是1
static bool TestSoftmax() {
  Tensor x({3}, 0.0f);
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  Tensor y = Softmax(x);
  MINI_LLAMA_ASSERT_NEAR(y[0], 0.09003057f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[1], 0.24472847f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[2], 0.66524096f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[0] + y[1] + y[2], 1.0f, 1e-5f);
  return true;
}

//三个0一样大，每个都应该是1/3
static bool TestSoftmaxUniform() {
  Tensor x({3}, 0.0f);
  Tensor y = Softmax(x);
  for (int i = 0; i < 3; ++i) {
    MINI_LLAMA_ASSERT_NEAR(y[i], 1.0f / 3.0f, 1e-5f);
  }
  return true;
}

//Softmax只接受一维，二维应该报错
static bool TestSoftmaxWrongDim() {
  Tensor x({2, 3}, 0.0f);
  try {
    Softmax(x);
    MINI_LLAMA_ASSERT_FAIL("expected exception for Softmax on 2D tensor");
  } catch (const std::runtime_error&) {
  }
  return true;
}

//Silu(0)=0，Silu(1)大约是0.731
static bool TestSilu() {
  Tensor x({1}, 0.0f);
  Tensor y = Silu(x);
  MINI_LLAMA_ASSERT_NEAR(y[0], 0.0f, 1e-5f);
  x[0] = 1.0f;
  y = Silu(x);
  MINI_LLAMA_ASSERT_NEAR(y[0], 0.7310586f, 1e-5f);
  return true;
}

//对应位置相乘：1*4、2*5、3*6
static bool TestElementwiseMul() {
  Tensor a({3}, 0.0f);
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  Tensor b({3}, 0.0f);
  b[0] = 4;
  b[1] = 5;
  b[2] = 6;
  Tensor y = ElementwiseMul(a, b);
  MINI_LLAMA_ASSERT_NEAR(y[0], 4.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[1], 10.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[2], 18.0f, 1e-5f);
  return true;
}

//SwiGLU就是先对gate做SiLU，再和up逐个相乘
static bool TestSwiGlu() {
  Tensor gate({3}, 0.0f);
  gate[0] = 0;
  gate[1] = 1;
  gate[2] = 2;
  Tensor up({3}, 0.0f);
  up[0] = 1;
  up[1] = 2;
  up[2] = 3;
  Tensor y = SwiGlu(gate, up);
  MINI_LLAMA_ASSERT_NEAR(y[0], 0.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[1], 0.7310586f * 2.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[2], 1.7615942f * 3.0f, 1e-4f);
  return true;
}

//最大值5出现了两次，取第一次的下标1
static bool TestArgMax() {
  Tensor x({5}, 0.0f);
  x[0] = 1;
  x[1] = 5;
  x[2] = 3;
  x[3] = 5;
  x[4] = 2;
  MINI_LLAMA_ASSERT_EQ(ArgMax(x), 1);  //第一次出现的最大值下标
  return true;
}

//启动时把上面的算子测试登记进极简框架
static struct OpsTestRegistrar {
  OpsTestRegistrar() {
    RegisterTest("matmul_identity", TestMatmulIdentity);
    RegisterTest("matmul_simple", TestMatmulSimple);
    RegisterTest("matmul_rectangular", TestMatmulRectangular);
    RegisterTest("matmul_shape_mismatch", TestMatmulShapeMismatch);
    RegisterTest("linear_1d", TestLinear1d);
    RegisterTest("rmsnorm_unit", TestRmsNormUnit);
    RegisterTest("rmsnorm_with_weight", TestRmsNormWithWeight);
    RegisterTest("rmsnorm_shape_mismatch", TestRmsNormShapeMismatch);
    RegisterTest("softmax", TestSoftmax);
    RegisterTest("softmax_uniform", TestSoftmaxUniform);
    RegisterTest("softmax_wrong_dim", TestSoftmaxWrongDim);
    RegisterTest("silu", TestSilu);
    RegisterTest("elementwise_mul", TestElementwiseMul);
    RegisterTest("swiglu", TestSwiGlu);
    RegisterTest("argmax", TestArgMax);
  }
} g_ops_test_registrar;
