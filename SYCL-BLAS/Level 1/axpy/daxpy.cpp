#include "../../blas.hpp"
#include "daxpy.hpp"

namespace sblas
{
    void daxpy(uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q, bool Async)
    {
        daxpyFunction::daxpyClass(N, alpha, x, incX, y, incY, q);
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