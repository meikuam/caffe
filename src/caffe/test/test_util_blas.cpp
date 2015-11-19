#ifndef CPU_ONLY  // CPU-GPU test

#include <cstring>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename TypeParam>
class GemmTest : public ::testing::Test {};

TYPED_TEST_CASE(GemmTest, TestDtypes);

TYPED_TEST(GemmTest, TestGemmCPUGPU) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  Blob<Dtype,Mtype> A(1, 1, 2, 3);
  Blob<Dtype,Mtype> B(1, 1, 3, 4);
  Blob<Dtype,Mtype> C(1, 1, 2, 4);
  Dtype data[12] = {1, 2, 3, 4,
                    5, 6, 7, 8,
                    9, 10, 11, 12};
  Dtype A_reshape_data[6] = {1, 4, 2, 5,
                             3, 6};
  Dtype B_reshape_data[12] = {1, 5, 9, 2,
                              6, 10, 3, 7,
                              11, 4, 8, 12};
  Dtype result[8] = {38, 44, 50, 56,
                     83, 98, 113, 128};
  caffe_copy(6, data, A.mutable_cpu_data());
  caffe_copy(12, data, B.mutable_cpu_data());

  if (sizeof(Dtype) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    // [1, 2, 3; 4 5 6] * [1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(result[i], C.cpu_data()[i]);
    }
    caffe_gpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(result[i], C.cpu_data()[i]);
    }

    // Test when we have a transposed A
    A.Reshape(1, 1, 3, 2);
    caffe_copy(6, A_reshape_data, A.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(result[i], C.cpu_data()[i]);
    }
    caffe_gpu_gemm<Dtype,Mtype>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(result[i], C.cpu_data()[i]);
    }

    // Test when we have a transposed A and a transposed B too
    B.Reshape(1, 1, 4, 3);
    caffe_copy(12, B_reshape_data, B.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(result[i], C.cpu_data()[i]);
    }
    caffe_gpu_gemm<Dtype,Mtype>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(result[i], C.cpu_data()[i]);
    }

    // Test when we have a transposed B
    A.Reshape(1, 1, 2, 3);
    caffe_copy(6, data, A.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(result[i], C.cpu_data()[i]);
    }
    caffe_gpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(result[i], C.cpu_data()[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GemmTest, TestGemmCPUGPUbeta1) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  Blob<Dtype,Mtype> A(1, 1, 3, 2);
  Blob<Dtype,Mtype> B(1, 1, 2, 1);
  Blob<Dtype,Mtype> C(1, 1, 3, 1);
  Dtype data[6] = {1, 2,
                   3, 4,
                   5, 6};
  Dtype result[3] = {5, 11, 17};
  caffe_copy(6, data, A.mutable_cpu_data());
  caffe_copy(2, data, B.mutable_cpu_data());

  if (sizeof(Dtype) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    caffe_copy(3, result, C.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, 3, 1, 2, 1.,
        A.cpu_data(), B.cpu_data(), 1., C.mutable_cpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(result[i] * 2., C.cpu_data()[i]);
    }
    caffe_copy(3, result, C.mutable_cpu_data());
    caffe_gpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, 3, 1, 2, 1.,
        A.gpu_data(), B.gpu_data(), 1., C.mutable_gpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(result[i] * 2., C.cpu_data()[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GemmTest, TestGemvCPUGPU) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  Blob<Dtype,Mtype> A(1, 1, 2, 3);
  Blob<Dtype,Mtype> x(1, 1, 1, 3);
  Blob<Dtype,Mtype> y(1, 1, 1, 2);
  Dtype data[6] = {1, 2, 3,
                   4, 5, 6};
  Dtype result_2[2] = {14, 32};
  Dtype result_3[3] = {9, 12, 15};
  caffe_copy(6, data, A.mutable_cpu_data());
  caffe_copy(3, data, x.mutable_cpu_data());

  if (sizeof(Dtype) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    caffe_cpu_gemv<Dtype,Mtype>(CblasNoTrans, 2, 3, 1., A.cpu_data(),
        x.cpu_data(), 0., y.mutable_cpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(result_2[i], y.cpu_data()[i]);
    }
    caffe_gpu_gemv<Dtype,Mtype>(CblasNoTrans, 2, 3, 1., A.gpu_data(),
        x.gpu_data(), 0., y.mutable_gpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(result_2[i], y.cpu_data()[i]);
    }

    // Test transpose case
    caffe_copy(2, data, y.mutable_cpu_data());
    caffe_cpu_gemv<Dtype,Mtype>(CblasTrans, 2, 3, 1., A.cpu_data(),
        y.cpu_data(), 0., x.mutable_cpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(result_3[i], x.cpu_data()[i]);
    }
    caffe_gpu_gemv<Dtype,Mtype>(CblasTrans, 2, 3, 1., A.gpu_data(),
        y.gpu_data(), 0., x.mutable_gpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(result_3[i], x.cpu_data()[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GemmTest, TestGemvCPUGPU2) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  Blob<Dtype,Mtype> A(1, 1, 3, 2);
  Blob<Dtype,Mtype> x(1, 1, 1, 2);
  Blob<Dtype,Mtype> y(1, 1, 1, 3);
  Dtype data[6] = {1, 2,
                   3, 4,
                   5, 6};
  Dtype result_3[3] = {5, 11, 17};
  Dtype result_2[2] = {22, 28};
  caffe_copy(6, data, A.mutable_cpu_data());
  caffe_copy(2, data, x.mutable_cpu_data());

  if (sizeof(Dtype) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    caffe_cpu_gemv<Dtype,Mtype>(CblasNoTrans, 3, 2, 1., A.cpu_data(),
        x.cpu_data(), 0., y.mutable_cpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(result_3[i], y.cpu_data()[i]);
    }
    caffe_gpu_gemv<Dtype,Mtype>(CblasNoTrans, 3, 2, 1., A.gpu_data(),
        x.gpu_data(), 0., y.mutable_gpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(result_3[i], y.cpu_data()[i]);
    }

    // Test transpose case
    caffe_copy(3, data, y.mutable_cpu_data());
    caffe_cpu_gemv<Dtype,Mtype>(CblasTrans, 3, 2, 1., A.cpu_data(),
        y.cpu_data(), 0., x.mutable_cpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(result_2[i], x.cpu_data()[i]);
    }
    caffe_gpu_gemv<Dtype,Mtype>(CblasTrans, 3, 2, 1., A.gpu_data(),
        y.gpu_data(), 0., x.mutable_gpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(result_2[i], x.cpu_data()[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe

#endif  // CPU_ONLY
