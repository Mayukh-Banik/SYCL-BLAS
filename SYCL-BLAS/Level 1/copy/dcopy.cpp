#include "../../blas.hpp"
#include "dcopy.hpp"
#include "COPY.hpp"


namespace sblas
{
    void dcopy(uint64_t N, double* x, int incX, double* y, int incY, sycl::queue q, bool Async, int index)
    {
        functionSelector<dcopyFunc::funcTable>(functionIndex<NUMBER_OF_FUNCTIONS>(index), N, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace dcopyFunc
{
    void defaultDcopyFunction(uint64_t N, double* x, int incX, double* y, int incY, sycl::queue q)
    {
        copy(N, x, incX, y, incY, q);
    }
}