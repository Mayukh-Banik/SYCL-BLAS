#include "../../blas.hpp"
#include "caxpy.hpp"

namespace sblas
{
    void caxpy(uint64_t N, float alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q, bool Async)
    {
        caxpyFunction::caxpyClass(N, alpha, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace caxpyFunction
{
    void defaultCaxpyFunction(uint64_t N, float alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q)
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