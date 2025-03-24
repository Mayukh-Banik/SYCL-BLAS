#include "../../blas.hpp"
#include "daxpy.hpp"

namespace sblas
{
    void daxpy(uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q, bool Async, int index)
    {
        functionSelector<daxpyFunc::funcTable>(index == -1 ? daxpyFunc::defaultFunction : index, N, alpha, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace daxpyFunc
{
    void defaultDaxpyFunction(uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q)
    {
        simpleAxpy(N, alpha, x, incX, y, incY, q);
    }
}