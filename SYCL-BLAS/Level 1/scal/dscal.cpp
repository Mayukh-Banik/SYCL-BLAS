#include "../../blas.hpp"
#include "dscal.hpp"

namespace sblas
{
    void dscal(uint64_t N, double alpha , double* x, int incX, sycl::queue q, bool Async, int index)
    {
        functionSelector<dscalFunc::funcTable>(index == -1 ? dscalFunc::defaultFunction : index, N, alpha, x, incX, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace dscalFunc
{
    void defaultDscalFunction(uint64_t N, double alpha, double* x, int incX, sycl::queue q)
    {
        scal(N, alpha, x, incX, q);
    }
}