#include "../../blas.hpp"
#include "saxpy.hpp"

namespace sblas
{
    void saxpy(uint64_t N, float alpha, float *x, int incX, float *y, int incY, sycl::queue q, bool Async, int index)
    {
        functionSelector<saxpyFunc::funcTable>(functionIndex<NUMBER_OF_FUNCTIONS>(index), N, alpha, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace saxpyFunc
{
    void defaultSaxpyFunction(uint64_t N, float alpha, float *x, int incX, float *y, int incY, sycl::queue q)
    {
        simpleAxpy(N, alpha, x, incX, y, incY, q);
    }
}