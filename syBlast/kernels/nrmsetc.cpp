#include "../syBlast.hpp"
#include "nrmsetc.hpp"

namespace syBlast
{

    using namespace templatedFunctions;

    float snrm2(
        const uint64_t N,
        const float *x,
        const int incx,
        sycl::queue &q)
    {
        return nrm2(N, x, incx, q);
    }

    float dnrm2(
        const uint64_t N,
        const double *x,
        const int incx,
        sycl::queue &q)
    {
        return nrm2(N, x, incx, q);
    }

    sycl::Complex<float> scnrm2(
        const uint64_t N,
        const sycl::Complex<float> *x,
        const int incx,
        sycl::queue &q)
    {
        return nrm2(N, x, incx, q);
    }

    sycl::Complex<double> dznrm2(
        const uint64_t N,
        const sycl::Complex<double> *x,
        const int incx,
        sycl::queue &q)
    {
        return nrm2(N, x, incx, q);
    }

    float sasum(
        const uint64_t N,
        const float *x,
        const int incx,
        sycl::queue &q)
    {
        return asum(N, x, incx, q);
    }

    float dasum(const uint64_t N, const double *x, const int incx, sycl::queue &q)
    {
        return asum(N, x, incx, q);
    }

    sycl::Complex<float> scasum(const uint64_t N, const sycl::Complex<float> *x, const int incx, sycl::queue &q)
    {
        return asum(N, x, incx, q);
    }

    sycl::Complex<double> dzasum(const uint64_t N, const sycl::Complex<double> *x, const int incx, sycl::queue &q)
    {
        return asum(N, x, incx, q);
    }

}