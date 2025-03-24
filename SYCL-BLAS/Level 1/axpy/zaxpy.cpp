#include "../../blas.hpp"
#include "zaxpy.hpp"

namespace sblas
{
    void zaxpy(uint64_t N, double alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q, bool Async)
    {
        functionSelector<zaxpyFunction::funcTable>(zaxpyFunction::defaultFunction, N, alpha, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace zaxpyFunction
{
    void defaultZaxpyFunction(uint64_t N, double alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q)
    {
        simpleAxpy(N, alpha, x, incX, y, incY, q);
    }
}