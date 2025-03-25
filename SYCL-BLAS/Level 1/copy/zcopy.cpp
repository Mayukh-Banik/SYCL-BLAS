#include "../../blas.hpp"
#include "zcopy.hpp"
#include "COPY.hpp"


namespace sblas
{
    void zcopy(uint64_t N, std::complex<double>* x, int incX, std::complex<double>* y, int incY, sycl::queue q, bool Async, int index)
    {
        functionSelector<zcopyFunc::funcTable>(functionIndex<NUMBER_OF_FUNCTIONS>(index), N, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace zcopyFunc
{
    void defaultZcopyFunction(uint64_t N, std::complex<double>* x, int incX, std::complex<double>* y, int incY, sycl::queue q)
    {
        copy(N, x, incX, y, incY, q);
    }
}