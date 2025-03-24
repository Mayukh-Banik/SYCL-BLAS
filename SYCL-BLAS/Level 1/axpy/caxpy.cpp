#include "../../blas.hpp"
#include "caxpy.hpp"

namespace sblas
{
    void caxpy(uint64_t N, float alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q, bool Async)
    {
        functionSelector<caxpyFunction::funcTable>(caxpyFunction::defaultFunction, N, alpha, x, incX, y, incY, q);
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
        simpleAxpy(N, alpha, x, incX, y, incY, q);
    }
}