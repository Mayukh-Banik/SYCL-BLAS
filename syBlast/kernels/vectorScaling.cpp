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

    void scopy(
        const uint64_t N,
        const float *x,
        const int incx,
        float *y,
        const int incy,
        sycl::queue &q)
    {
        copy(N, x, incx, y, incy, q);
    }

    void dcopy(
        const uint64_t N,
        const double *x,
        const int incx,
        double *y,
        const int incy,
        sycl::queue &q)
    {
        copy(N, x, incx, y, incy, q);
    }

    void ccopy(
        const uint64_t N,
        const std::complex<float> *x,
        const int incx,
        std::complex<float> *y,
        const int incy,
        sycl::queue &q)
    {
        copy(N, x, incx, y, incy, q);
    }

    void zcopy(
        const uint64_t N,
        const std::complex<double> *x,
        const int incx,
        std::complex<double> *y,
        const int incy,
        sycl::queue &q)
    {
        copy(N, x, incx, y, incy, q);
    }

    void sscal(
        const uint64_t N,
        const float alpha,
        float *x,
        const int incx,
        sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }

    void dscal(
        const uint64_t N,
        const double alpha,
        double *x,
        const int incx,
        sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }

    void cscal(
        const uint64_t N,
        const float alpha,
        std::complex<float> *x,
        const int incx,
        sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }

    void zscal(
        const uint64_t N,
        const double alpha,
        std::complex<double> *x,
        const int incx,
        sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }

    void csscal(
        const uint64_t N,
        const std::complex<float> alpha,
        std::complex<float> *x,
        const int incx,
        sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }

    void zdscal(
        const uint64_t N,
        const std::complex<double> alpha,
        std::complex<double> *x,
        const int incx,
        sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }

    void sswap(
        const uint64_t N,
        float *x,
        const int incx,
        float *y,
        const int incy,
        sycl::queue &q)
    {
        swap(N, x, incx, y, incy, q);
    }

    void dswap(
        const uint64_t N,
        double *x,
        const int incx,
        double *y,
        const int incy,
        sycl::queue &q)
    {
        swap(N, x, incx, y, incy, q);
    }

    void cswap(
        const uint64_t N,
        std::complex<float> *x,
        const int incx,
        std::complex<float> *y,
        const int incy,
        sycl::queue &q)
    {
        swap(N, x, incx, y, incy, q);
    }

    void zswap(
        const uint64_t N,
        std::complex<double> *x,
        const int incx,
        std::complex<double> *y,
        const int incy,
        sycl::queue &q)
    {
        swap(N, x, incx, y, incy, q);
    }
}