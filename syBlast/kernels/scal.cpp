#include "../syBlast_template.hpp"
#include "../syBlast.hpp"

namespace syBlast
{
    using namespace templatedFunctions;

    void sscal(const uint64_t N, const float alpha, float *x, const int incx, sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }

    void dscal(const uint64_t N, const double alpha, double *x, const int incx, sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }

    void cscal(const uint64_t N, const float alpha, std::complex<float> *x, const int incx, sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }

    void zscal(const uint64_t N, const double alpha, std::complex<double> *x, const int incx, sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }

    void csscal(const uint64_t N, const std::complex<float> alpha, std::complex<float> *x, const int incx, sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }

    void zdscal(const uint64_t N, const std::complex<double> alpha, std::complex<double> *x, const int incx, sycl::queue &q)
    {
        scal(N, alpha, x, incx, q);
    }
}