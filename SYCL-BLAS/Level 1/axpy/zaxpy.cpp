#include "../../blas.hpp"
#include "zaxpy.hpp"

namespace sblas
{
    void zaxpy(uint64_t N, double alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q, bool Async)
    {
        functionSelector<zaxpyFunc::funcTable>(zaxpyFunc::defaultFunction, N, alpha, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace zaxpyFunc
{
    void defaultZaxpyFunction(uint64_t N, double alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q)
    {
        simpleAxpy(N, alpha, x, incX, y, incY, q);
    }
}