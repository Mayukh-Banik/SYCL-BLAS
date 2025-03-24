#include "../../blas.hpp"
#include "zaxpy.hpp"

namespace sblas
{
    void zaxpy(uint64_t N, double alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q, bool Async)
    {
        zaxpyFunction::zaxpyClass(N, alpha, x, incX, y, incY, q);
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