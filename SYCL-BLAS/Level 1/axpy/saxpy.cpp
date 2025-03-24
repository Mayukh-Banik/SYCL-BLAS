#include "../../blas.hpp"
#include "saxpy.hpp"

namespace sblas
{
    void saxpy(uint64_t N, float alpha, float *x, int incX, float *y, int incY, sycl::queue q, bool Async)
    {
        functionSelector<saxpyFunction::funcTable>(saxpyFunction::defaultFunction, N, alpha, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace saxpyFunction
{
    void defaultSaxpyFunction(uint64_t N, float alpha, float *x, int incX, float *y, int incY, sycl::queue q)
    {
        simpleAxpy(N, alpha, x, incX, y, incY, q);
    }
}