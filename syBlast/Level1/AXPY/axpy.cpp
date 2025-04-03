#include "AXPY.hpp"
#include "../../blas.hpp"

#define SWITCH_CASE_MACRO(param, ...) \
    switch (param)                          \
    {                                       \
    case 0:                                 \
        axpy(__VA_ARGS__);                  \
        break;                              \
    default:                                \
        axpy(__VA_ARGS__);                  \
        break;                              \
    }

namespace syBlast
{
    void saxpy(
        const uint64_t N,
        const float alpha,
        const float *x,
        const int incx,
        float *y,
        const int incy,
        sycl::queue q,
        database::Parameters p)
    {
        SWITCH_CASE_MACRO(p[0], N, alpha, x, incx, y, incy, q, p);
    }

    void daxpy(
        const uint64_t N,
        const double alpha,
        const double *x,
        const int incx,
        double *y,
        const int incy,
        sycl::queue q,
        database::Parameters p)
    {
        SWITCH_CASE_MACRO(p[0], N, alpha, x, incx, y, incy, q, p);
    }

    void caxpy(
        const uint64_t N,
        const std::complex<float> alpha,
        const std::complex<float> *x,
        const int incx,
        std::complex<float> *y,
        const int incy,
        sycl::queue q,
        database::Parameters p)
    {
        SWITCH_CASE_MACRO(p[0], N, alpha, x, incx, y, incy, q, p);
    }

    void zaxpy(
        const uint64_t N,
        const std::complex<double> alpha,
        const std::complex<double> *x,
        const int incx,
        std::complex<double> *y,
        const int incy,
        sycl::queue q,
        database::Parameters p)
    {
        SWITCH_CASE_MACRO(p[0], N, alpha, x, incx, y, incy, q, p);
    }
}
