#include "../syBlast_template.hpp"
#include "../syBlast.hpp"

namespace syBlast
{
    using namespace templatedFunctions;

    void sswap(
        const uint64_t N,
        const float *x,
        const int incx,
        float *y,
        const int incy,
        sycl::queue &q)
    {
        swap(N, x, incx, y, incy, q);
    }

    void dswap(
        const uint64_t N,
        const double *x,
        const int incx,
        double *y,
        const int incy,
        sycl::queue &q)
    {
        swap(N, x, incx, y, incy, q);
    }

    void cswap(
        const uint64_t N,
        const std::complex<float> *x,
        const int incx,
        std::complex<float> *y,
        const int incy,
        sycl::queue &q)
    {
        swap(N, x, incx, y, incy, q);
    }

    void zswap(
        const uint64_t N,
        const std::complex<double> *x,
        const int incx,
        std::complex<double> *y,
        const int incy,
        sycl::queue &q)
    {
        swap(N, x, incx, y, incy, q);
    }
}