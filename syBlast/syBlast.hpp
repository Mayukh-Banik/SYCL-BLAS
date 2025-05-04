#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include "syComplex.hpp"

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
    void caxpy(const uint64_t N, const sycl::Complex<float> alpha, const sycl::Complex<float> *x, const int incx, sycl::Complex<float> *y, const int incy, sycl::queue &q);

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
    void zaxpy(const uint64_t N, const sycl::Complex<double> alpha, const sycl::Complex<double> *x, const int incx, sycl::Complex<double> *y, const int incy, sycl::queue &q);

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
    void cscal(const uint64_t N, const float alpha, sycl::Complex<float> *x, const int incx, sycl::queue &q);

    /**
     * x := alpha * x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param q     SYCL queue for execution
     */
    void zscal(const uint64_t N, const double alpha, sycl::Complex<double> *x, const int incx, sycl::queue &q);

    /**
     * x := alpha * x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param q     SYCL queue for execution
     */
    void csscal(const uint64_t N, const sycl::Complex<float> alpha, sycl::Complex<float> *x, const int incx, sycl::queue &q);

    /**
     * x := alpha * x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param q     SYCL queue for execution
     */
    void zdscal(const uint64_t N, const sycl::Complex<double> alpha, sycl::Complex<double> *x, const int incx, sycl::queue &q);

    /**
     * y := x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void scopy(const uint64_t N, const float *x, const int incx, float *y, const int incy, sycl::queue &q);

    /**
     * y := x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void dcopy(const uint64_t N, const double *x, const int incx, double *y, const int incy, sycl::queue &q);

    /**
     * y := x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void ccopy(const uint64_t N, const sycl::Complex<float> *x, const int incx, sycl::Complex<float> *y, const int incy, sycl::queue &q);

    /**
     * y := x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void zcopy(const uint64_t N, const sycl::Complex<double> *x, const int incx, sycl::Complex<double> *y, const int incy, sycl::queue &q);

    /**
     * y <--> x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void sswap(const uint64_t N, float *x, const int incx, float *y, const int incy, sycl::queue &q);

    /**
     * y <--> x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void dswap(const uint64_t N, double *x, const int incx, double *y, const int incy, sycl::queue &q);

    /**
     * y <--> x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void cswap(const uint64_t N, sycl::Complex<float> *x, const int incx, sycl::Complex<float> *y, const int incy, sycl::queue &q);

    /**
     * y <--> x
     *
     * @param N     Number of elements in the vectors
     * @param alpha Scalar multiplier for x
     * @param x     Pointer to the input vector x
     * @param incx  Stride between elements in x
     * @param y     Pointer to the output vector y (also input for accumulation)
     * @param incy  Stride between elements in y
     * @param q     SYCL queue for execution
     */
    void zswap(const uint64_t N, sycl::Complex<double> *x, const int incx, sycl::Complex<double> *y, const int incy, sycl::queue &q);

    float sdot(const uint64_t N, const float *x, const int incx, const float *y, const int incy, sycl::queue &q);

    double ddot(const uint64_t N, const double *x, const int incx, const double *y, const int incy, sycl::queue &q);

    sycl::Complex<float> cdotu(const uint64_t N, const sycl::Complex<float> *x, const int incx, const sycl::Complex<float> *y, const int incy, sycl::queue &q);

    sycl::Complex<double> zdotu(const uint64_t N, const sycl::Complex<double> *x, const int incx, const sycl::Complex<double> *y, const int incy, sycl::queue &q);

    sycl::Complex<float> cdotc(const uint64_t N, const sycl::Complex<float> *x, const int incx, const sycl::Complex<float> *y, const int incy, sycl::queue &q);

    sycl::Complex<double> zdotc(const uint64_t N, const sycl::Complex<double> *x, const int incx, const sycl::Complex<double> *y, const int incy, sycl::queue &q);

    float sdsdot(const uint64_t N, const float alpha, const float *x, const int incx, const float *y, const int incy, sycl::queue &q);

    float dsdot(const uint64_t N, const double *x, const int incx, const float *y, const int incy, sycl::queue &q);

    float snrm2(const uint64_t N, const float* x, const int incx, sycl::queue& q);

    float dnrm2(const uint64_t N, const double* x, const int incx, sycl::queue& q);

    sycl::Complex<float> scnrm2(const uint64_t N, const sycl::Complex<float>* x, const int incx, sycl::queue& q);

    sycl::Complex<double> dznrm2(const uint64_t N, const sycl::Complex<double>* x, const int incx, sycl::queue& q);

    float sasum(const uint64_t N, const float* x, const int incx, sycl::queue& q);

    float dasum(const uint64_t N, const double* x, const int incx, sycl::queue& q);

    sycl::Complex<float> scasum(const uint64_t N, const sycl::Complex<float>* x, const int incx, sycl::queue& q);
    
    sycl::Complex<double> dzasum(const uint64_t N, const sycl::Complex<double>* x, const int incx, sycl::queue& q);
}