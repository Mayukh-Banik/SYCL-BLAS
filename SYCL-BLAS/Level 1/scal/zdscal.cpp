#include "../../blas.hpp"
#include "zdscal.hpp"

namespace sblas
{
    void zdscal(uint64_t N, double alpha , std::complex<double>* x, int incX, sycl::queue q, bool Async, int index)
    {
        functionSelector<zdscalFunc::funcTable>(functionIndex<NUMBER_OF_FUNCTIONS>(index), N, alpha, x, incX, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace zdscalFunc
{
    void defaultZdscalFunction(uint64_t N, double alpha, std::complex<double>* x, int incX, sycl::queue q)
    {
        scal(N, alpha, x, incX, q);
    }
}