#include "../../blas.hpp"
#include "sscal.hpp"

namespace sblas
{
    void sscal(uint64_t N, float alpha , float* x, int incX, sycl::queue q, bool Async, int index)
    {
        functionSelector<SscalFunc::funcTable>(index == -1 ? SscalFunc::defaultFunction : index, N, alpha, x, incX, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace SscalFunc
{
    void defaultSscalFunction(uint64_t N, float alpha, float* x, int incX, sycl::queue q)
    {
        scal(N, alpha, x, incX, q);
    }
}