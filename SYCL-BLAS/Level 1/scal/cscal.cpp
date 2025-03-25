#include "../../blas.hpp"
#include "cscal.hpp"

namespace sblas
{
    void cscal(uint64_t N, std::complex<float> alpha, std::complex<float>* x, int incX, sycl::queue q, bool Async, int index)
    {
        functionSelector<cscalFunc::funcTable>(functionIndex<NUMBER_OF_FUNCTIONS>(index), N, alpha, x, incX, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace cscalFunc
{
    void defaultCscalFunction(uint64_t N, std::complex<float> alpha, std::complex<float>* x, int incX, sycl::queue q)
    {
        scal(N, alpha, x, incX, q);
    }
}