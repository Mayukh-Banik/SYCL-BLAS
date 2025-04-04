#pragma once

#include "../../blas.hpp"

template <typename A, typename B = A>
void scal(
    const uint64_t N,
    const A alpha,
    B *x,
    const int incx,
    sycl::queue &q,
    syBlast::database::Parameters p)
{
    q.submit(
        [&](sycl::handler &handler)
        {
            handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                 { x[i[0]] *= alpha; });
        });
}