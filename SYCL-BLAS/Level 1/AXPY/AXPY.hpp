#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include "../../Database/database_structures.hpp"


template <typename T>
void axpy(
    const uint64_t N,
    const T alpha,
    const T *x,
    const int incx,
    T *y,
    const int incy,
    sycl::queue q,
    syBlast::database::Parameters p)
{
    q.submit(
        [&](sycl::handler &handler)
        {
            handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                 {
           const int64_t Xindex = i[0] * incx;
           const int64_t Yindex = i[0] * incy;
           y[Yindex] += alpha * x[Xindex]; });
        });
}