#include "SCAL.hpp"
#include "../../blas.hpp"

#define SWITCH_CASE_MACRO(param, ...) \
    switch (param)                          \
    {                                       \
    case 0:                                 \
        scal(__VA_ARGS__);                  \
        break;                              \
    default:                                \
        scal(__VA_ARGS__);                  \
        break;                              \
    }

namespace syBlast
{
    void sscal(
        const uint64_t N,
        const float alpha,
        float *x,
        const int incx,
        sycl::queue& q,
        database::Parameters p)
    {
        SWITCH_CASE_MACRO(p[0], N, alpha, x, incx, q, p);
    }

    void dscal(
        const uint64_t N,
        const double alpha,
        double *x,
        const int incx,
        sycl::queue& q,
        database::Parameters p)
    {
        SWITCH_CASE_MACRO(p[0], N, alpha, x, incx, q, p);
    }

    void cscal(
        const uint64_t N,
        const std::complex<float> alpha,
        std::complex<float> *x,
        const int incx,
        sycl::queue& q,
        database::Parameters p)
    {
        SWITCH_CASE_MACRO(p[0], N, alpha, x, incx, q, p);
    }

    void zscal(
        const uint64_t N,
        const std::complex<double> alpha,
        std::complex<double> *x,
        const int incx,
        sycl::queue& q,
        database::Parameters p)
    {
        SWITCH_CASE_MACRO(p[0], N, alpha, x, incx, q, p);
    }

    void csscal(
        const uint64_t N,
        const float alpha,
        std::complex<float> *x,
        const int incx,
        sycl::queue& q,
        database::Parameters p)
    {
        SWITCH_CASE_MACRO(p[0], N, alpha, x, incx, q, p);
    }

    void zsscal(
        const uint64_t N,
        const double alpha,
        std::complex<double> *x,
        const int incx,
        sycl::queue& q,
        database::Parameters p)
    {
        SWITCH_CASE_MACRO(p[0], N, alpha, x, incx, q, p);
    }
}