#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <complex>
#include "Database/database.hpp"

// #if defined(_WIN32) || defined(_WIN64)
// #define EXPORT_SYMBOL __declspec(dllexport)
// #else
// #define EXPORT_SYMBOL __attribute__((visibility("default")))
// #endif

/**
 * @mainpage syBlast Documentation
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
 * @ref syBlast
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
    void saxpy(const uint64_t N, const float alpha, const float *x, const int incx, float *y, const int incy, sycl::queue &q, database::Parameters p = parameters::FunctionParameters());
    /**
     * @brief Performs the DAXPY operation: y = alpha * x + y
     *
     * This function computes the double-precision A·X Plus Y (DAXPY) operation,
     * which scales a vector `x` by a scalar `alpha` and adds the result to a vector `y`.
     *
     * @param N The number of elements in the vectors `x` and `y`.
     * @param alpha The scalar multiplier for the vector `x`.
     * @param x Pointer to the input vector `x`.
     * @param incx The increment between consecutive elements of `x`.
     * @param y Pointer to the input/output vector `y`.
     * @param incy The increment between consecutive elements of `y`.
     * @param q SYCL queue to execute the operation.
     * @param p (Optional) Parameters from the database for tuning the operation.
     *           Defaults to the parameters for the DAXPY operation in the database.
     */
    void daxpy(const uint64_t N, const double alpha, const double *x, const int incx, double *y, const int incy, sycl::queue &q, database::Parameters p = parameters::FunctionParameters());
    /**
     * @brief Performs the CAXPY operation: y = alpha * x + y
     *
     * This function computes the single-precision complex A·X Plus Y (CAXPY) operation,
     * which scales a complex vector `x` by a complex scalar `alpha` and adds the result to a complex vector `y`.
     *
     * @param N The number of elements in the vectors `x` and `y`.
     * @param alpha The complex scalar multiplier for the vector `x`.
     * @param x Pointer to the input complex vector `x`.
     * @param incx The increment between consecutive elements of `x`.
     * @param y Pointer to the input/output complex vector `y`.
     * @param incy The increment between consecutive elements of `y`.
     * @param q SYCL queue to execute the operation.
     * @param p (Optional) Parameters from the database for tuning the operation.
     *           Defaults to the parameters for the CAXPY operation in the database.
     */
    void caxpy(const uint64_t N, const std::complex<float> alpha, const std::complex<float> *x, const int incx, std::complex<float> *y, const int incy, sycl::queue &q, database::Parameters p = parameters::FunctionParameters());
    /**
     * @brief Performs the ZAXPY operation: y = alpha * x + y
     *
     * This function computes the double-precision complex A·X Plus Y (ZAXPY) operation,
     * which scales a complex vector `x` by a complex scalar `alpha` and adds the result to a complex vector `y`.
     *
     * @param N The number of elements in the vectors `x` and `y`.
     * @param alpha The complex scalar multiplier for the vector `x`.
     * @param x Pointer to the input complex vector `x`.
     * @param incx The increment between consecutive elements of `x`.
     * @param y Pointer to the input/output complex vector `y`.
     * @param incy The increment between consecutive elements of `y`.
     * @param q SYCL queue to execute the operation.
     * @param p (Optional) Parameters from the database for tuning the operation.
     *           Defaults to the parameters for the ZAXPY operation in the database.
     */
    void zaxpy(const uint64_t N, const std::complex<double> alpha, const std::complex<double> *x, const int incx, std::complex<double> *y, const int incy, sycl::queue &q, database::Parameters p = parameters::FunctionParameters());
    /**
     * @brief Performs the SSCAL operation: x = alpha * x
     *
     * This function scales a single-precision vector `x` by a scalar `alpha`.
     *
     * @param N The number of elements in the vector `x`.
     * @param alpha The scalar multiplier for the vector `x`.
     * @param x Pointer to the input/output vector `x`.
     * @param incx The increment between consecutive elements of `x`.
     * @param q SYCL queue to execute the operation.
     * @param p (Optional) Parameters from the database for tuning the operation.
     *           Defaults to the parameters for the SSCAL operation in the database.
     */
    void sscal(const uint64_t N, const float alpha, float *x, const int incx, sycl::queue &q, database::Parameters p = parameters::FunctionParameters());
    /**
     * @brief Performs the DSCAL operation: x = alpha * x
     *
     * This function scales a double-precision vector `x` by a scalar `alpha`.
     *
     * @param N The number of elements in the vector `x`.
     * @param alpha The scalar multiplier for the vector `x`.
     * @param x Pointer to the input/output vector `x`.
     * @param incx The increment between consecutive elements of `x`.
     * @param q SYCL queue to execute the operation.
     * @param p (Optional) Parameters from the database for tuning the operation.
     *           Defaults to the parameters for the DSCAL operation in the database.
     */
    void dscal(const uint64_t N, const double alpha, double *x, const int incx, sycl::queue &q, database::Parameters p = parameters::FunctionParameters());
    /**
     * @brief Performs the CSCAL operation: x = alpha * x
     *
     * This function scales a single-precision complex vector `x` by a complex scalar `alpha`.
     *
     * @param N The number of elements in the vector `x`.
     * @param alpha The complex scalar multiplier for the vector `x`.
     * @param x Pointer to the input/output complex vector `x`.
     * @param incx The increment between consecutive elements of `x`.
     * @param q SYCL queue to execute the operation.
     * @param p (Optional) Parameters from the database for tuning the operation.
     *           Defaults to the parameters for the CSCAL operation in the database.
     */
    void cscal(const uint64_t N, const std::complex<float> alpha, std::complex<float> *x, const int incx, sycl::queue &q, database::Parameters p = parameters::FunctionParameters());
    /**
     * @brief Performs the ZSCAL operation: x = alpha * x
     *
     * This function scales a double-precision complex vector `x` by a complex scalar `alpha`.
     *
     * @param N The number of elements in the vector `x`.
     * @param alpha The complex scalar multiplier for the vector `x`.
     * @param x Pointer to the input/output complex vector `x`.
     * @param incx The increment between consecutive elements of `x`.
     * @param q SYCL queue to execute the operation.
     * @param p (Optional) Parameters from the database for tuning the operation.
     *           Defaults to the parameters for the ZSCAL operation in the database.
     */
    void zscal(const uint64_t N, const std::complex<double> alpha, std::complex<double> *x, const int incx, sycl::queue &q, database::Parameters p = parameters::FunctionParameters());
    /**
     * @brief Performs the CSSCAL operation: x = alpha * x
     *
     * This function scales a single-precision complex vector `x` by a real scalar `alpha`.
     *
     * @param N The number of elements in the vector `x`.
     * @param alpha The real scalar multiplier for the vector `x`.
     * @param x Pointer to the input/output complex vector `x`.
     * @param incx The increment between consecutive elements of `x`.
     * @param q SYCL queue to execute the operation.
     * @param p (Optional) Parameters from the database for tuning the operation.
     *           Defaults to the parameters for the CSSCAL operation in the database.
     */
    void csscal(const uint64_t N, const float alpha, std::complex<float> *x, const int incx, sycl::queue &q, database::Parameters p = parameters::FunctionParameters());
    /**
     * @brief Performs the ZSSCAL operation: x = alpha * x
     *
     * This function scales a double-precision complex vector `x` by a real scalar `alpha`.
     *
     * @param N The number of elements in the vector `x`.
     * @param alpha The real scalar multiplier for the vector `x`.
     * @param x Pointer to the input/output complex vector `x`.
     * @param incx The increment between consecutive elements of `x`.
     * @param q SYCL queue to execute the operation.
     * @param p (Optional) Parameters from the database for tuning the operation.
     *           Defaults to the parameters for the ZSSCAL operation in the database.
     */
    void zsscal(const uint64_t N, const double alpha, std::complex<double> *x, const int incx, sycl::queue &q, database::Parameters p = parameters::FunctionParameters());
















    
}
