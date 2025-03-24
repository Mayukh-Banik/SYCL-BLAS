#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>
#include <complex>

/**
 * @namespace sblas
 * 
 * Location for all NetLib conformant C style functions from library SYCL-BLAS. 
 * All functions located here will follow NetLib style naming and function parameters, except for the last 2 parameters.
 * The first is sycl::queue q, which is the default device for the sycl code to run on, and the device that
 * all the raw memory pointers is expected to be on.
 * The second is bool Async = false with Async = false meaning computation will be done asynchronously,
 * otherwise q.wait() will be called.
 */
namespace sblas
{
    void saxpy(uint64_t N, float alpha, float *x, int incX, float *y, int incY, sycl::queue q = sycl::queue(), bool Async = false);
    void daxpy(uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q = sycl::queue(), bool Async = false);
    void caxpy(uint64_t N, float alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q = sycl::queue(), bool Async = false);
    void zaxpy(uint64_t N, double alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q = sycl::queue(), bool Async = false);

}