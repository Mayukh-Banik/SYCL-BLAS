#include "../../blas.hpp"
#include "csscal.hpp"

namespace sblas
{
    void csscal(uint64_t N, float alpha, std::complex<float>* x, int incX, sycl::queue q, bool Async, int index)
    {
        functionSelector<csscalFunc::funcTable>(index == -1 ? csscalFunc::defaultFunction : index, N, alpha, x, incX, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace csscalFunc
{
    void defaultCsscalFunction(uint64_t N, float alpha, std::complex<float>* x, int incX, sycl::queue q)
    {
        scal(N, alpha, x, incX, q);
    }
}