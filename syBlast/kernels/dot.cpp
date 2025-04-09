#include "../syBlast_template.hpp"
#include "../syBlast.hpp"

namespace syBlast
{
    using namespace templatedFunctions;

    float sdot(
        const uint64_t N,
        const float *x,
        const int incx,
        const float *y,
        const int incy,
        sycl::queue &q)
    {
        return dot(N, x, incx, y, incy, q);
    }

    double ddot(
        const uint64_t N,
        const double *x,
        const int incx,
        const double *y,
        const int incy,
        sycl::queue &q)
    {
        return dot(N, x, incx, y, incy, q);
    }

    std::complex<float> cdotu(
        const uint64_t N,
        const std::complex<float> *x,
        const int incx,
        const std::complex<float> *y,
        const int incy,
        sycl::queue &q)
    {
        return dot(N, x, incx, y, incy, q);
    }

    std::complex<double> zdotu(
        const uint64_t N,
        const std::complex<double> *x,
        const int incx,
        const std::complex<double> *y,
        const int incy,
        sycl::queue &q)
    {
        return dot(N, x, incx, y, incy, q);
    }

    std::complex<float> cdotc(
        const uint64_t N,
        const std::complex<float> *x,
        const int incx,
        const std::complex<float> *y,
        const int incy,
        sycl::queue &q)
    {
        return dotc(N, x, incx, y, incy, q);
    }

    std::complex<double> zdotc(
        const uint64_t N,
        const std::complex<double> *x,
        const int incx,
        const std::complex<double> *y,
        const int incy,
        sycl::queue &q)
    {
        return dotc(N, x, incx, y, incy, q);
    }

    float sdsdot(
        const uint64_t N,
        const float alpha, 
        const float *x,
        const int incx,
        const float *y,
        const int incy,
        sycl::queue &q)
    {
        return alpha + dot(N, x, incx, y, incy, q);
    }

    float dsdot(
        const uint64_t N,
        const double *x,
        const int incx,
        const float *y,
        const int incy,
        sycl::queue &q)
    {
        return dot(N, x, incx, y, incy, q);
    }
}