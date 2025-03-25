#include "../../blas.hpp"
#include "zscal.hpp"

namespace sblas
{
    void zdscal(uint64_t N, std::complex<double> alpha , std::complex<double>* x, int incX, sycl::queue q, bool Async, int index)
    {
        functionSelector<zscalFunc::funcTable>(index == -1 ? zscalFunc::defaultFunction : index, N, alpha, x, incX, q);
        if (!Async)
        {
            q.wait();
        }
    }
}

namespace zscalFunc
{
    void defaultZscalFunction(uint64_t N, std::complex<double> alpha, std::complex<double>* x, int incX, sycl::queue q)
    {
        scal(N, alpha, x, incX, q);
    }
}