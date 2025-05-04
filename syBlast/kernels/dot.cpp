#include "../syBlast.hpp"
#include "dot.hpp"

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

    sycl::Complex<float> cdotu(
        const uint64_t N,
        const sycl::Complex<float> *x,
        const int incx,
        const sycl::Complex<float> *y,
        const int incy,
        sycl::queue &q)
    {
        return dot(N, x, incx, y, incy, q);
    }

    sycl::Complex<double> zdotu(
        const uint64_t N,
        const sycl::Complex<double> *x,
        const int incx,
        const sycl::Complex<double> *y,
        const int incy,
        sycl::queue &q)
    {
        return dot(N, x, incx, y, incy, q);
    }

    sycl::Complex<float> cdotc(
        const uint64_t N,
        const sycl::Complex<float> *x,
        const int incx,
        const sycl::Complex<float> *y,
        const int incy,
        sycl::queue &q)
    {
        return dotc(N, x, incx, y, incy, q);
    }

    sycl::Complex<double> zdotc(
        const uint64_t N,
        const sycl::Complex<double> *x,
        const int incx,
        const sycl::Complex<double> *y,
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