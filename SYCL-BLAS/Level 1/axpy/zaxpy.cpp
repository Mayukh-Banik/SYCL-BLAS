#include "../../blas.hpp"
#include "zaxpy.hpp"

namespace sblas
{
    void zaxpy(uint64_t N, std::complex<double> alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q, bool Async, int index)
    {
        functionSelector<zaxpyFunc::funcTable>(index == -1 ? zaxpyFunc::defaultFunction : index, N, alpha, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace zaxpyFunc
{
    void defaultZaxpyFunction(uint64_t N, std::complex<double> alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q)
    {
        simpleAxpy(N, alpha, x, incX, y, incY, q);
    }
}