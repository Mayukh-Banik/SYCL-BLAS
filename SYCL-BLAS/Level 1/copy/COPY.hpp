#pragma once

#include <complex>
#include <sycl/sycl.hpp>
#include <cstdint>
#include "../../Setup/UsefulFunctions.hpp"

template <typename A, typename B = A>
void copy(uint64_t N, A* x, int incX, B* y, int incY, sycl::queue q)
{
    q.submit(
        [&](sycl::handler& handler)
        {
            handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
            {
                const int64_t xIndex = i[0] * incX;
                const int64_t yIndex = i[0] * incY;
                y[yIndex] = static_cast<B>(x[xIndex]);
            });
        }
    );
}