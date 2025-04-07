#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <complex>

namespace syBlast
{

    void saxpy(const uint64_t N, const float alpha, const float *x, const int incx, float *y, const int incy, sycl::queue &q);

    void daxpy(const uint64_t N, const double alpha, const double *x, const int incx, double *y, const int incy, sycl::queue &q);

    void caxpy(const uint64_t N, const std::complex<float> alpha, const std::complex<float> *x, const int incx, std::complex<float> *y, const int incy, sycl::queue &q);

    void zaxpy(const uint64_t N, const std::complex<double> alpha, const std::complex<double> *x, const int incx, std::complex<double> *y, const int incy, sycl::queue &q);

    void sscal(const uint64_t N, const float alpha, float *x, const int incx, sycl::queue &q);

    void dscal(const uint64_t N, const double alpha, double *x, const int incx, sycl::queue &q);

    void cscal(const uint64_t N, const float alpha, std::complex<float> *x, const int incx, sycl::queue &q);

    void zscal(const uint64_t N, const double alpha, std::complex<double> *x, const int incx, sycl::queue &q);

    void csscal(const uint64_t N, const std::complex<float> alpha, std::complex<float> *x, const int incx, sycl::queue &q);

    void zdscal(const uint64_t N, const std::complex<double> alpha, std::complex<double> *x, const int incx, sycl::queue &q);

    void scopy(const uint64_t N, const float *x, const int incx, float *y, const int incy, sycl::queue &q);

    void dcopy(const uint64_t N, const double *x, const int incx, double *y, const int incy, sycl::queue &q);

    void ccopy(const uint64_t N, const std::complex<float> *x, const int incx, std::complex<float> *y, const int incy, sycl::queue &q);

    void zcopy(const uint64_t N, const std::complex<double> *x, const int incx, std::complex<double> *y, const int incy, sycl::queue &q);

    void sswap(const uint64_t N, const float *x, const int incx, float *y, const int incy, sycl::queue &q);

    void dswap(const uint64_t N, const double *x, const int incx, double *y, const int incy, sycl::queue &q);

    void cswap(const uint64_t N, const std::complex<float> *x, const int incx, std::complex<float> *y, const int incy, sycl::queue &q);

    void zswap(const uint64_t N, const std::complex<double> *x, const int incx, std::complex<double> *y, const int incy, sycl::queue &q);
}