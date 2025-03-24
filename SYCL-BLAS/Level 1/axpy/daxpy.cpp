#include "../../blas.hpp"
#include "daxpy.hpp"

namespace sblas
{
    void daxpy(uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q, bool Async)
    {
        functionSelector<daxpyFunction::funcTable>(daxpyFunction::defaultFunction, N, alpha, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace daxpyFunction
{
    void defaultDaxpyFunction(uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q)
    {
        simpleAxpy(N, alpha, x, incX, y, incY, q);
    }
}