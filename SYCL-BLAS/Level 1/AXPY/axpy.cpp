#include "AXPY.hpp"

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