#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <type_traits>
#include "syComplex.hpp"

namespace syBlast::templatedFunctions
{
    template <typename A, typename B = A>
    B nrm2(
        const uint64_t N,
        const A *x,
        const int incx,
        sycl::queue &q)
    {
        A *temp = sycl::malloc_shared<A>(sizeof(A), q);
        *temp = A(0);
        q.submit(
            [&](sycl::handler &handler)
            {
                auto reducer = sycl::reduction(x, sycl::plus<A>());
                handler.parallel_for(sycl::range<1>(N), reducer, [=](sycl::id<1> i, auto &sum)
                                     { sum += sycl::pown(x[i[0] * incx], 2); });
            });
        q.wait();
        A ret = sycl::sqrt(temp[0]);
        sycl::free(temp);
        if constexpr (std::is_same_v<A, B>)
        {
            return ret;
        }
        else
        {
            return (B)ret;
        }
    }

    template <typename A, typename B = A>
    B asum(
        const uint64_t N,
        const A *x,
        const int incx,
        sycl::queue &q)
    {
        A *temp = sycl::malloc_shared<A>(sizeof(A), q);
        *temp = A(0);
        q.submit(
            [&](sycl::handler &handler)
            {
                auto reducer = sycl::reduction(x, sycl::plus<A>());
                handler.parallel_for(sycl::range<1>(N), reducer, [=](sycl::id<1> i, auto &sum)
                                     { sum += sycl::pown(x[i[0] * incx], 2); });
            });
        q.wait();
        A ret = sycl::sqrt(temp[0]);
        sycl::free(temp);
        if constexpr (std::is_same_v<A, B>)
        {
            return ret;
        }
        else
        {
            return (B)ret;
        }
    }

    template <typename A, typename B = A>
    B asum(
        const uint64_t N,
        const sycl::Complex<A> *x,
        const int incx,
        sycl::queue &q)
    {
        A *temp = sycl::malloc_shared<A>(sizeof(A), q);
        *temp = A(0);
        q.submit(
            [&](sycl::handler &handler)
            {
                
                auto reducer = sycl::reduction(x, sycl::plus<A>());
                handler.parallel_for(sycl::range<1>(N), reducer, [=](sycl::id<1> i, auto &sum)
                                     {
                                        sycl::Complex<A>& val = x[incx * i[0]];
                                        sum += sycl::abs(val.real()) + sycl::abs(val.real()); });
            });
        q.wait();
        A ret = sycl::sqrt(temp[0]);
        sycl::free(temp);
        if constexpr (std::is_same_v<A, B>)
        {
            return ret;
        }
        else
        {
            return (B)ret;
        }
    }
}