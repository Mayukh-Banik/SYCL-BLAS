#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <complex>
#include "Database/database.hpp"

#if defined(_WIN32) || defined(_WIN64)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif

/**
 * @mainpage SYCL-BLAS Documentation
 *
 * @section intro_sec Introduction
 *
 * An open source implementation of BLAS in SYCL with Tuners for multiple platforms.
 *
 * @section install_sec Installation
 *
 * 1. SYCL Compiler
 * 2. CMake
 * 3. SYCL capable device (everything except AMD CPUS and new AMD GPUs).
 *
 * @section usage_sec Usage
 *
 * @ref sblas
 */


/**
 * @namespace syBlast
 *
 * Location for all NetLib conformant C style functions from library SYCL-BLAS.
 * All functions located here will follow NetLib style naming and function parameters.
 */
namespace syBlast
{
    /**
     * @brief Performs the SAXPY operation: y = alpha * x + y
     *
     * This function computes the single-precision A·X Plus Y (SAXPY) operation,
     * which scales a vector `x` by a scalar `alpha` and adds the result to a vector `y`.
     *
     * @param N The number of elements in the vectors `x` and `y`.
     * @param alpha The scalar multiplier for the vector `x`.
     * @param x Pointer to the input vector `x`.
     * @param incx The increment between consecutive elements of `x`.
     * @param y Pointer to the input/output vector `y`.
     * @param incy The increment between consecutive elements of `y`.
     * @param q (Optional) SYCL queue to execute the operation. Defaults to a new SYCL queue.
     * @param p (Optional) Parameters from the database for tuning the operation.
     *           Defaults to the parameters for the SAXPY operation in the database.
     */
    void saxpy(const uint64_t N, const float alpha, const float *x, const int incx, float *y, const int incy, sycl::queue &q, database::Parameters p = database::FuncParamDB[BLAS_ENUM_NAMES::SAXPY]);

    void daxpy(const uint64_t N, const double alpha, const double *x, const int incx, double *y, const int incy, sycl::queue q = sycl::queue(), database::Parameters p = database::FuncParamDB[BLAS_ENUM_NAMES::DAXPY]);

    void caxpy(const uint64_t N, const std::complex<float> alpha, const std::complex<float> *x, const int incx, std::complex<float> *y, const int incy, sycl::queue q = sycl::queue(), database::Parameters p = database::FuncParamDB[BLAS_ENUM_NAMES::CAXPY]);

    void zaxpy(const uint64_t N, const std::complex<double> alpha, const std::complex<double> *x, const int incx, std::complex<double> *y, const int incy, sycl::queue q = sycl::queue(), database::Parameters p = database::FuncParamDB[BLAS_ENUM_NAMES::ZAXPY]);

}
