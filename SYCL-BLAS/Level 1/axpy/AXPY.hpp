#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <utility>
#include "../UsefulFunctions.hpp"

#define DEFAULT_FUNCTION_INDEX 0

template <typename A, typename B = A, typename C = A>
void simpleAxpy(uint64_t N, A alpha, B *x, int incX, C *y, int incY, sycl::queue q)
{
    q.submit(
        [&](sycl::handler &handler)
        {
            handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                 {
           const int64_t Xindex = i[0] * incX;
           const int64_t Yindex = i[0] * incY;
           y[Yindex] += (C) alpha * (C) x[Xindex]; });
        });
}

template <typename A, typename B = A, typename C = A>
void simpleAxpy(uint64_t N, A alpha, std::complex<B> *x, int incX, std::complex<C> *y, int incY, sycl::queue q)
{
    q.submit([&](sycl::handler &handler)
    {
        handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                             {
            const int64_t Xindex = i[0] * incX;
            const int64_t Yindex = i[0] * incY;
            y[Yindex] = static_cast<C>(static_cast<B>(alpha) * x[Xindex]);
        });
    });
}

template <typename A, typename B = A, typename C = A>
void simpleAxpy(uint64_t N, std::complex<A> alpha, std::complex<B> *x, int incX, std::complex<C> *y, int incY, sycl::queue q)
{
    q.submit([&](sycl::handler &handler)
    {
        handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                             {
            const int64_t Xindex = i[0] * incX;
            const int64_t Yindex = i[0] * incY;
            y[Yindex] = static_cast<C>(static_cast<B>(alpha) * x[Xindex]);
        });
    });
}
