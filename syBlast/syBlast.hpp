#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <complex>

namespace syBlast
{

    /**
     * y := alpha * x + y
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void saxpy(const uint64_t N, const float alpha, const float *x, const int incx, float *y, const int incy, sycl::queue &q);

    /**
     * y := alpha * x + y
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void daxpy(const uint64_t N, const double alpha, const double *x, const int incx, double *y, const int incy, sycl::queue &q);

    /**
     * y := alpha * x + y
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void caxpy(const uint64_t N, const std::complex<float> alpha, const std::complex<float> *x, const int incx, std::complex<float> *y, const int incy, sycl::queue &q);

    /**
     * y := alpha * x + y
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void zaxpy(const uint64_t N, const std::complex<double> alpha, const std::complex<double> *x, const int incx, std::complex<double> *y, const int incy, sycl::queue &q);

    /**
     * x := alpha * x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param q     SYCL queue for execution
     */
    void sscal(const uint64_t N, const float alpha, float *x, const int incx, sycl::queue &q);

    /**
     * x := alpha * x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param q     SYCL queue for execution
     */
    void dscal(const uint64_t N, const double alpha, double *x, const int incx, sycl::queue &q);

    /**
     * x := alpha * x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param q     SYCL queue for execution
     */
    void cscal(const uint64_t N, const float alpha, std::complex<float> *x, const int incx, sycl::queue &q);

    /**
     * x := alpha * x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param q     SYCL queue for execution
     */
    void zscal(const uint64_t N, const double alpha, std::complex<double> *x, const int incx, sycl::queue &q);

    /**
     * x := alpha * x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param q     SYCL queue for execution
     */
    void csscal(const uint64_t N, const std::complex<float> alpha, std::complex<float> *x, const int incx, sycl::queue &q);

    /**
     * x := alpha * x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param q     SYCL queue for execution
     */
    void zdscal(const uint64_t N, const std::complex<double> alpha, std::complex<double> *x, const int incx, sycl::queue &q);

    /**
     * @brief Copies a single-precision vector: y := x
     */
    void scopy(const uint64_t N, const float *x, const int incx, float *y, const int incy, sycl::queue &q);

    /**
     * @brief Copies a double-precision vector: y := x
     */
    void dcopy(const uint64_t N, const double *x, const int incx, double *y, const int incy, sycl::queue &q);

    /**
     * @brief Copies a complex single-precision vector: y := x
     */
    void ccopy(const uint64_t N, const std::complex<float> *x, const int incx, std::complex<float> *y, const int incy, sycl::queue &q);

    /**
     * @brief Copies a complex double-precision vector: y := x
     */
    void zcopy(const uint64_t N, const std::complex<double> *x, const int incx, std::complex<double> *y, const int incy, sycl::queue &q);

    /**
     * @brief Swaps two single-precision vectors: x <-> y
     */
    void sswap(const uint64_t N, const float *x, const int incx, float *y, const int incy, sycl::queue &q);

    /**
     * @brief Swaps two double-precision vectors: x <-> y
     */
    void dswap(const uint64_t N, const double *x, const int incx, double *y, const int incy, sycl::queue &q);

    /**
     * @brief Swaps two complex single-precision vectors: x <-> y
     */
    void cswap(const uint64_t N, const std::complex<float> *x, const int incx, std::complex<float> *y, const int incy, sycl::queue &q);

    /**
     * @brief Swaps two complex double-precision vectors: x <-> y
     */
    void zswap(const uint64_t N, const std::complex<double> *x, const int incx, std::complex<double> *y, const int incy, sycl::queue &q);

}