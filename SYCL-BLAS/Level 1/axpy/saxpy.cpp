#include "../../blas.hpp"
#include "saxpy.hpp"

namespace sblas
{
    void saxpy(uint64_t N, float alpha, float *x, int incX, float *y, int incY, sycl::queue q, bool Async)
    {
        saxpyFunction::saxpyClass(N, alpha, x, incX, y, incY, q);
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
        q.submit(
            [&](sycl::handler &handler)
            {
                handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                     {
               int64_t Xindex = i[0] * incX;
               int64_t Yindex = i[0] * incY;
               y[Yindex] += alpha * x[Xindex]; });
            });
    }
}