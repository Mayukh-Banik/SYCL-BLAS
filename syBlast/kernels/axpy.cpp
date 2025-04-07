#include "../syBlast_template.hpp"
#include "../syBlast.hpp"

namespace syBlast
{

using namespace templatedFunctions;
    void saxpy(
        const uint64_t N,
        const float alpha,
        const float *x,
        const int incx,
        float *y,
        const int incy,
        sycl::queue &q)
    {
        axpy(N, alpha, x, incx, y, incy, q);
    }

    void daxpy(
        const uint64_t N,
        const double alpha,
        const double *x,
        const int incx,
        double *y,
        const int incy,
        sycl::queue &q)
    {
        axpy(N, alpha, x, incx, y, incy, q);
    }

    void caxpy(
        const uint64_t N,
        const std::complex<float> alpha,
        const std::complex<float> *x,
        const int incx,
        std::complex<float> *y,
        const int incy,
        sycl::queue &q)
    {
        axpy(N, alpha, x, incx, y, incy, q);
    }

    void zaxpy(
        const uint64_t N,
        const std::complex<double> alpha,
        const std::complex<double> *x,
        const int incx,
        std::complex<double> *y,
        const int incy,
        sycl::queue &q)
    {
        axpy(N, alpha, x, incx, y, incy, q);
    }
}