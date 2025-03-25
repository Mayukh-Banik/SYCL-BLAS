#include "../../blas.hpp"
#include "ccopy.hpp"
#include "COPY.hpp"


namespace sblas
{
    void ccopy(uint64_t N, std::complex<float>* x, int incX, std::complex<float>* y, int incY, sycl::queue q, bool Async, int index)
    {
        functionSelector<ccopyFunc::funcTable>(functionIndex<NUMBER_OF_FUNCTIONS>(index), N, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace ccopyFunc
{
    void defaultCcopyFunction(uint64_t N, std::complex<float>* x, int incX, std::complex<float>* y, int incY, sycl::queue q)
    {
        copy(N, x, incX, y, incY, q);
    }
}