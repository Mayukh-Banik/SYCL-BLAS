#include "AXPY.hpp"
#include "../../blas_c.hpp"

namespace syBlast
{
    void saxpy(
        const uint64_t N,
        const float alpha,
        const float *x,
        const int incx,
        float *y,
        const int incy,
        sycl::queue q,
        database::Parameters p)
    {
        switch (p[0])
        {
        case 0:
            axpy(N, alpha, x, incx, y, incy, q, p);
        default:
            axpy(N, alpha, x, incx, y, incy, q, p);
            break;
        }
    }
}

EXPORT_SYMBOL void saxpy(int n, float a, const float *x, int incx, float *y, int incy)
{
    syBlast::saxpy(n, a, x, incx, y, incy);
}
