#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>
#include <complex>

/**
 * @namespace sblas
 *
 * Location for all NetLib conformant C style functions from library SYCL-BLAS.
 * All functions located here will follow NetLib style naming and function parameters, except for the last 3 parameters.
 * The first is sycl::queue q, which is the default device for the sycl code to run on, and the device that
 * all the raw memory pointers is expected to be on.
 * The second is bool Async = false with Async = false meaning computation will be done asynchronously,
 * otherwise q.wait() will be called.
 * The third is int index = -1. index refers to specific function to be used. This is not meant to ever be called,
 * except for tuning/testing.
 */
namespace sblas
{
    /**
     * @brief Performs the SAXPY operation: y = alpha * x + y
     *
     * Computes the sum of a scalar `alpha` multiplied by a vector `x` and added to a vector `y`.
     *
     * @param N The number of elements in the vectors `x` and `y`.
     * @param alpha The scalar multiplier for the vector `x`.
     * @param x Pointer to the input vector `x`. Expected to be allocated with sycl::malloc*.
     * @param incX The increment for the elements of `x`.
     * @param y Pointer to the input/output vector `y`. Expected to be allocated with sycl::malloc*.
     * @param incY The increment for the elements of `y`.
     * @param q The SYCL queue for execution.
     * @param Async If true, the computation is performed asynchronously.
     * @param index Index for testing/tuning purposes (default is -1).
     */
    void saxpy(uint64_t N, float alpha, float *x, int incX, float *y, int incY, sycl::queue q = sycl::queue(), bool Async = false, int index = -1);

    /**
     * @brief Performs the DAXPY operation: y = alpha * x + y
     *
     * Computes the sum of a scalar `alpha` multiplied by a vector `x` and added to a vector `y`.
     *
     * @param N The number of elements in the vectors `x` and `y`.
     * @param alpha The scalar multiplier for the vector `x`.
     * @param x Pointer to the input vector `x`. Expected to be allocated with sycl::malloc*.
     * @param incX The increment for the elements of `x`.
     * @param y Pointer to the input/output vector `y`. Expected to be allocated with sycl::malloc*.
     * @param incY The increment for the elements of `y`.
     * @param q The SYCL queue for execution.
     * @param Async If true, the computation is performed asynchronously.
     * @param index Index for testing/tuning purposes (default is -1).
     */
    void daxpy(uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q = sycl::queue(), bool Async = false, int index = -1);

    /**
     * @brief Performs the CAXPY operation: y = alpha * x + y
     *
     * Computes the sum of a scalar `alpha` multiplied by a vector `x` and added to a vector `y`,
     * where `x` and `y` are complex numbers.
     *
     * @param N The number of elements in the vectors `x` and `y`.
     * @param alpha The scalar multiplier for the vector `x`.
     * @param x Pointer to the input complex vector `x`. Expected to be allocated with sycl::malloc*.
     * @param incX The increment for the elements of `x`.
     * @param y Pointer to the input/output complex vector `y`. Expected to be allocated with sycl::malloc*.
     * @param incY The increment for the elements of `y`.
     * @param q The SYCL queue for execution.
     * @param Async If true, the computation is performed asynchronously.
     * @param index Index for testing/tuning purposes (default is -1).
     */
    void caxpy(uint64_t N, float alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q = sycl::queue(), bool Async = false, int index = -1);

    /**
     * @brief Performs the ZAXPY operation: y = alpha * x + y
     *
     * Computes the sum of a scalar `alpha` multiplied by a vector `x` and added to a vector `y`,
     * where `x` and `y` are complex numbers.
     *
     * @param N The number of elements in the vectors `x` and `y`.
     * @param alpha The scalar multiplier for the vector `x`.
     * @param x Pointer to the input complex vector `x`. Expected to be allocated with sycl::malloc*.
     * @param incX The increment for the elements of `x`.
     * @param y Pointer to the input/output complex vector `y`. Expected to be allocated with sycl::malloc*.
     * @param incY The increment for the elements of `y`.
     * @param q The SYCL queue for execution.
     * @param Async If true, the computation is performed asynchronously.
     * @param index Index for testing/tuning purposes (default is -1).
     */
    void zaxpy(uint64_t N, double alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q = sycl::queue(), bool Async = false, int index = -1);

}