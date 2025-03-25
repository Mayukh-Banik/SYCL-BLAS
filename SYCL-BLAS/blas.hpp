#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>
#include <complex>

/** 
 * @mainpage MyProject Documentation
 *
 * @section intro_sec Introduction
 *
 * This is an example project demonstrating how to add documentation to the main page.
 *
 * @section install_sec Installation
 *
 * 1. Clone the repository
 * 2. Run `make`
 * 3. Execute `./myproject`
 *
 * @section usage_sec Usage
 *
 * Here's how you use this program:
 * @code
 * MyClass obj;
 * obj.doSomething();
 * @endcode
 *
 * @author Your Name
 * @version 1.0
 * @date March 2025
 */


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

    /**
     * @brief Scales a complex single-precision vector by a complex scalar using SYCL.
     *
     * @param N The number of elements in the vector.
     * @param alpha The complex scalar multiplier.
     * @param x Pointer to the input/output complex vector.
     * @param incX Stride between consecutive elements in x.
     * @param q The SYCL queue to execute the operation (default: new SYCL queue).
     * @param Async Whether to execute asynchronously (currently unused).
     * @param index Optional index parameter (default: -1, currently unused).
     */
    void cscal(uint64_t N, std::complex<float> alpha, std::complex<float> *x, int incX, sycl::queue q = sycl::queue(), bool Async = false, int index = -1);

    /**
     * @brief Scales a complex single-precision vector by a real scalar using SYCL.
     *
     * @param N The number of elements in the vector.
     * @param alpha The real scalar multiplier.
     * @param x Pointer to the input/output complex vector.
     * @param incX Stride between consecutive elements in x.
     * @param q The SYCL queue to execute the operation (default: new SYCL queue).
     * @param Async Whether to execute asynchronously (currently unused).
     * @param index Optional index parameter (default: -1, currently unused).
     */
    void csscal(uint64_t N, float alpha, std::complex<float> *x, int incX, sycl::queue q = sycl::queue(), bool Async = false, int index = -1);

    /**
     * @brief Scales a double-precision vector by a real scalar using SYCL.
     *
     * @param N The number of elements in the vector.
     * @param alpha The real scalar multiplier.
     * @param x Pointer to the input/output double-precision vector.
     * @param incX Stride between consecutive elements in x.
     * @param q The SYCL queue to execute the operation (default: new SYCL queue).
     * @param Async Whether to execute asynchronously (currently unused).
     * @param index Optional index parameter (default: -1, currently unused).
     */
    void dscal(uint64_t N, double alpha, double *x, int incX, sycl::queue q = sycl::queue(), bool Async = false, int index = -1);

    /**
     * @brief Scales a single-precision vector by a real scalar using SYCL.
     *
     * @param N The number of elements in the vector.
     * @param alpha The real scalar multiplier.
     * @param x Pointer to the input/output single-precision vector.
     * @param incX Stride between consecutive elements in x.
     * @param q The SYCL queue to execute the operation (default: new SYCL queue).
     * @param Async Whether to execute asynchronously (currently unused).
     * @param index Optional index parameter (default: -1, currently unused).
     */
    void sscal(uint64_t N, float alpha, float *x, int incX, sycl::queue q = sycl::queue(), bool Async = false, int index = -1);

    /**
     * @brief Scales a complex double-precision vector by a real scalar using SYCL.
     *
     * @param N The number of elements in the vector.
     * @param alpha The real scalar multiplier.
     * @param x Pointer to the input/output complex double-precision vector.
     * @param incX Stride between consecutive elements in x.
     * @param q The SYCL queue to execute the operation (default: new SYCL queue).
     * @param Async Whether to execute asynchronously (currently unused).
     * @param index Optional index parameter (default: -1, currently unused).
     */
    void zdscal(uint64_t N, double alpha, std::complex<double> *x, int incX, sycl::queue q = sycl::queue(), bool Async = false, int index = -1);

    /**
     * @brief Scales a complex double-precision vector by a complex scalar using SYCL.
     *
     * @param N The number of elements in the vector.
     * @param alpha The complex scalar multiplier.
     * @param x Pointer to the input/output complex double-precision vector.
     * @param incX Stride between consecutive elements in x.
     * @param q The SYCL queue to execute the operation (default: new SYCL queue).
     * @param Async Whether to execute asynchronously (currently unused).
     * @param index Optional index parameter (default: -1, currently unused).
     */
    void zdscal(uint64_t N, std::complex<double> alpha, std::complex<double> *x, int incX, sycl::queue q = sycl::queue(), bool Async = false, int index = -1);

}