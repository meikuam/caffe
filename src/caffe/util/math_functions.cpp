#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float,float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double,double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

#ifndef CPU_ONLY
template<>
void caffe_cpu_gemm<float16,float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float16* A, const float16* B, const float beta,
    float16* C) {
  if (M <= 0 || N <= 0 || K <= 0) {
    return;
  }
  std::vector<float> a(M*K), b(K*N), c(M*N);
  caffe_cpu_convert(a.size(), A, &a.front());
  caffe_cpu_convert(b.size(), B, &b.front());
  caffe_cpu_convert(c.size(), C, &c.front());
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, &a.front(), lda, &b.front(),
      ldb, beta, &c.front(), N);
  caffe_cpu_convert(c.size(), &c.front(), C);
}
template<>
void caffe_cpu_gemm<float16,float16>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float16 alpha, const float16* A, const float16* B, const float16 beta,
    float16* C) {
  caffe_cpu_gemm(TransA, TransB, M, N, K, (float) alpha, A, B, (float) beta, C);
}
#endif

template <>
void caffe_cpu_gemv<float,float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double,double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

#ifndef CPU_ONLY
template <>
void caffe_cpu_gemv<float16,float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float16* A, const float16* x,
    const float beta, float16* y) {
  if (M <= 0 || N <= 0) {
    return;
  }
  const int lx = (TransA == CblasNoTrans) ? N : M;
  const int ly = (TransA == CblasNoTrans) ? M : N;
  std::vector<float> a(M*N), xv(lx), yv(ly);
  caffe_cpu_convert(a.size(), A, &a.front());
  caffe_cpu_convert(xv.size(), x, &xv.front());
  caffe_cpu_convert(yv.size(), y, &yv.front());
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, &a.front(), N, &xv.front(), 1, beta, &yv.front(), 1);
  caffe_cpu_convert(yv.size(), &yv.front(), y);
}
template <>
void caffe_cpu_gemv<float16,float16>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float16 alpha, const float16* A, const float16* x,
    const float16 beta, float16* y) {
  //  cblas_hgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
  caffe_cpu_gemv(TransA, M, N, (float) alpha, A, x, (float) beta, y);
}
#endif

template <>
void caffe_axpy<float,float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double,double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

#ifndef CPU_ONLY
// TODO Consider CUDA
template<>
void caffe_axpy<float16,float>(const int N, const float alpha, const float16* X, float16* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha * static_cast<float>(X[i]) + static_cast<float>(Y[i]);
  }
}
template<>
void caffe_axpy<float16,float16>(const int N, const float16 alpha, const float16* X, float16* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha * X[i] + Y[i];
  }
}
#endif

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

#ifndef CPU_ONLY
template void caffe_set<float16>(const int N, const float16 alpha, float16* Y);
#endif

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

#ifndef CPU_ONLY
template <>
void caffe_add_scalar(const int N, const float alpha, float16* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = Y[i] + alpha;
  }
}
template <>
void caffe_add_scalar(const int N, const float16 alpha, float16* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = Y[i] + alpha;
  }
}
#endif

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

#ifndef CPU_ONLY
template void caffe_copy<float16>(const int N, const float16* X, float16* Y);
#endif

template <>
void caffe_scal<float,float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double,double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

#ifndef CPU_ONLY
template <>
void caffe_scal<float16,float>(const int N, const float alpha, float16 *X) {
  for (int i = 0; i < N; ++i) {
    X[i] = alpha * X[i];
  }
}

template <>
void caffe_scal<float16,float16>(const int N, const float16 alpha, float16 *X) {
  // cblas_hscal(N, alpha, X, 1);
  caffe_scal(N, (float) alpha, X);
}
#endif

template <>
void caffe_cpu_axpby<float,float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double,double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

#ifndef CPU_ONLY
template <>
void caffe_cpu_axpby<float16,float>(const int N, const float alpha, const float16* X,
                             const float beta, float16* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha * X[i] + beta * Y[i];
  }
}
template <>
void caffe_cpu_axpby<float16,float16>(const int N, const float16 alpha, const float16* X,
            const float16 beta, float16* Y) {
  caffe_cpu_axpby(N, (float) alpha, X, (float) beta, Y);
}
#endif

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

#ifndef CPU_ONLY
template <>
void caffe_add<float16>(const int n, const float16* a, const float16* b,
    float16* y) {
  for (int i=0; i<n; ++i) {
    y[i] = a[i] + b[i];
  }
}
#endif

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

#ifndef CPU_ONLY
template <>
void caffe_sub<float16>(const int n, const float16* a, const float16* b,
    float16* y) {
  for (int i=0; i<n; ++i) {
    y[i] = a[i] - b[i];
  }
}
#endif

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

#ifndef CPU_ONLY
template <>
void caffe_mul<float16>(const int n, const float16* a, const float16* b,
    float16* y) {
  for (int i=0; i<n; ++i) {
    //  vhMul(n, a, b, y);
    y[i] = a[i] * b[i];
  }
}
#endif

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

#ifndef CPU_ONLY
template <>
void caffe_div<float16>(const int n, const float16* a, const float16* b, float16* y)
{
  //  vhDiv(n, a, b, y);
  for (int i=0; i<n; ++i) {
    y[i] = a[i] / b[i];
  }
}
#endif

template <>
void caffe_powx<float,float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double,double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

#ifndef CPU_ONLY
template <>
void caffe_powx<float16,float>(const int n, const float16* a, const float b, float16* y) {
  for (int i=0; i<n; i++) {
    y[i] = pow(static_cast<float>(a[i]), b);
  }
}
template <>
void caffe_powx<float16,float16>(const int n, const float16* a, const float16 b,
    float16* y) {
  //  vhPowx(n, a, b, y);
  caffe_powx(n, a, (float) b, y);
}
#endif

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

#ifndef CPU_ONLY
template <>
void caffe_sqr<float16>(const int n, const float16* a, float16* y) {
  vhSqr(n, a, y);
}
#endif

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

#ifndef CPU_ONLY
template <>
void caffe_exp<float16>(const int n, const float16* a, float16* y) {
  vhExp(n, a, y);
}
template <>
void caffe_log<float16>(const int n, const float16* a, float16* y) {
  vhLn(n, a, y);
}
#endif // ! CPU_ONLY

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

#ifndef CPU_ONLY
template <>
void caffe_abs<float16>(const int n, const float16* a, float16* y) {
  for (int i=0; i<n; i++) {
    y[i] = fabs(a[i]);
  }
}
#endif

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype, typename Mtype>
void caffe_rng_uniform(const int n, const Mtype a, const Mtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Mtype> random_distribution(a, caffe_nextafter<Mtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Mtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float,float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double,double>(const int n, const double a, const double b,
                               double* r);

#ifndef CPU_ONLY
template
void caffe_rng_uniform<float16,float>(const int n, const float a, const float b,
                               float16* r);
template<>
void caffe_rng_uniform<float16,float16>(const int n, const float16 a, const float16 b,
					float16* r) {
  caffe_rng_uniform<float16,float>(n, static_cast<const float>(a),
      static_cast<const float>(b), r);
}
#endif

template <typename Dtype, typename Mtype>
void caffe_rng_gaussian(const int n, const Mtype a,
                        const Mtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Mtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Mtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float,float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double,double>(const int n, const double mu,
                                const double sigma, double* r);

#ifndef CPU_ONLY
template
void caffe_rng_gaussian<float16,float>(const int n, const float mu,
                                const float sigma, float16* r);
template <>
void caffe_rng_gaussian<float16,float16>(const int n, const float16 mu,
					                      const float16 sigma, float16* r) {
  caffe_rng_gaussian<float16,float>(n, static_cast<const float>(mu),
      static_cast<const float>(sigma), r);
}
#endif

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);
template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

#ifndef CPU_ONLY
template<>
void caffe_rng_bernoulli<float16>(const int n, const float16 p, int* r) {
  caffe_rng_bernoulli(n, static_cast<const float>(p), r);
}
#endif

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);
template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

#ifndef CPU_ONLY
template<>
void caffe_rng_bernoulli<float16>(const int n, const float16 p, unsigned int* r) {
  caffe_rng_bernoulli(n, static_cast<const float>(p), r);
}
#endif

template <>
float caffe_cpu_strided_dot<float,float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double,double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

#ifndef CPU_ONLY
template <>
float caffe_cpu_strided_dot<float16,float>(const int n, const float16* x,
    const int incx, const float16 *y, const int incy) {
  float sum = 0.0f;
  int idx_x, idx_y;
  for (int i = 0; i < n; ++i) {
    idx_x = i*incx;
    idx_y = i*incy;
    sum += x[idx_x] * y[idx_y];
  }
  return sum;
}
// TODO Consider CUDA
template <>
float16 caffe_cpu_strided_dot<float16,float16>(const int n, const float16* x,
    const int incx, const float16 *y, const int incy) {
  return float16(caffe_cpu_strided_dot<float16,float>(n, x, incx, y, incy));
}
#endif

template <typename Dtype, typename Mtype>
Mtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot<Dtype,Mtype>(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float,float>(const int n, const float* x, const float* y);
template
double caffe_cpu_dot<double,double>(const int n, const double* x, const double* y);

#ifndef CPU_ONLY
template
float caffe_cpu_dot<float16,float>(const int n, const float16* x, const float16* y);
template
float16 caffe_cpu_dot<float16,float16>(const int n, const float16* x, const float16* y);
#endif

template <>
int caffe_cpu_hamming_distance<float>(const int n, const float* x,
                                  const float* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcount(static_cast<uint32_t>(x[i]) ^
                               static_cast<uint32_t>(y[i]));
  }
  return dist;
}

template <>
int caffe_cpu_hamming_distance<double>(const int n, const double* x,
                                   const double* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcountl(static_cast<uint64_t>(x[i]) ^
                                static_cast<uint64_t>(y[i]));
  }
  return dist;
}

#ifndef CPU_ONLY
template <>
int caffe_cpu_hamming_distance<float16>(const int n, const float16* x,
                                        const float16* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcount(static_cast<uint16_t>(x[i]) ^
                               static_cast<uint16_t>(y[i]));
  }
  return dist;
}
#endif

template <>
float caffe_cpu_asum<float,float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double,double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

#ifndef CPU_ONLY
template <>
float caffe_cpu_asum<float16,float>(const int n, const float16 *x) {
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += fabs(x[i]);
  }
  return sum;
}

template <>
float16 caffe_cpu_asum<float16,float16>(const int n, const float16 *x) {
  return float16(caffe_cpu_asum<float16,float>(n, x));
}
#endif

template <>
void caffe_cpu_scale<float,float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double,double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

#ifndef CPU_ONLY
template <>
void caffe_cpu_scale<float16,float>(const int n, const float alpha, const float16 *x,
    float16 *y) {
  for (int i=0; i<n; i++) {
    y[i] = alpha * x[i];
  }
}
template <>
void caffe_cpu_scale<float16,float16>(const int n, const float16 alpha, const float16 *x,
    float16 *y) {
  caffe_cpu_scale(n, (float) alpha, x, y);
}
#endif

}  // namespace caffe
