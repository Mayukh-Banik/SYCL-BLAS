#include "../../blas.hpp"
#include "caxpy.hpp"

namespace sblas
{
    void caxpy(uint64_t N, std::complex<float> alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q, bool Async, int index)
    {
        functionSelector<caxpyFunc::funcTable>(functionIndex<NUMBER_OF_FUNCTIONS>(index), N, alpha, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace caxpyFunc
{
    void defaultCaxpyFunction(uint64_t N, std::complex<float> alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q)
    {
        simpleAxpy(N, alpha, x, incX, y, incY, q);
    }
}