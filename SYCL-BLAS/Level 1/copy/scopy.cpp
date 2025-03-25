#include "../../blas.hpp"
#include "scopy.hpp"
#include "COPY.hpp"


namespace sblas
{
    void scopy(uint64_t N, float* x, int incX, float* y, int incY, sycl::queue q, bool Async, int index)
    {
        functionSelector<scopyFunc::funcTable>(functionIndex<NUMBER_OF_FUNCTIONS>(index), N, x, incX, y, incY, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace scopyFunc
{
    void defaultScopyFunction(uint64_t N, float* x, int incX, float* y, int incY, sycl::queue q)
    {
        copy(N, x, incX, y, incY, q);
    }
}