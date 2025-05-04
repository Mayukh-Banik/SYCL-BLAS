#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <type_traits>

namespace syBlast
{
    namespace templatedFunctions
    {

        template <typename A, typename B = A, typename C = A>
        C dot(
            const uint64_t N,
            const A *x,
            const int incx,
            const B *y,
            const int incy,
            sycl::queue &q)
        {
            C *sum_result = sycl::malloc_device<C>(1, q);
            *sum_result = C(0);
            q.submit(
                [&](sycl::handler &handler)
                {
                    auto reduction_sum = sycl::reduction(sum_result, sycl::plus<C>());
                    handler.parallel_for(
                        sycl::range<1>(N),
                        reduction_sum,
                        [=](sycl::id<1> i, auto &sum)
                        {
                            const int64_t Xindex = i[0] * incx;
                            const int64_t Yindex = i[0] * incy;
                            if constexpr (std::is_same_v<A, B> && std::is_same_v<B, C>)
                            {
                                sum.combine(static_cast<C>(x[Xindex] * y[Yindex]));
                            }
                            else
                            {
                                sum.combine(static_cast<C>(x[Xindex]) * static_cast<C>(y[Yindex]));
                            }
                        });
                });
            q.wait();
            C result;
            q.memcpy(&result, sum_result, sizeof(C));
            q.wait();
            sycl::free(sum_result, q);
            return result;
        }

        template <typename A, typename B = A, typename C = B>
        sycl::Complex<C> dotc(
            const uint64_t N,
            const sycl::Complex<A> *x,
            const int incx,
            const sycl::Complex<B> *y,
            const int incy,
            sycl::queue &q)
        {
            using ResultT = sycl::Complex<C>;
            ResultT *sum_result = sycl::malloc_device<ResultT>(1, q);
        
            ResultT zero = ResultT(0);
            q.memcpy(sum_result, &zero, sizeof(ResultT)).wait();
        
            q.submit(
                [&](sycl::handler &handler)
                {
                    auto reduction_sum = sycl::reduction(sum_result, sycl::plus<ResultT>());
                    handler.parallel_for(
                        sycl::range<1>(N),
                        reduction_sum,
                        [=](sycl::id<1> i, auto &sum)
                        {
                            const int64_t Xindex = i[0] * incx;
                            const int64_t Yindex = i[0] * incy;
                            auto xval = std::conj(x[Xindex]);
                            auto yval = y[Yindex];
                            sum.combine(static_cast<ResultT>(xval * yval));
                        });
                });
        
            q.wait();
        
            ResultT result;
            q.memcpy(&result, sum_result, sizeof(ResultT)).wait();
            sycl::free(sum_result, q);
            return result;
        }

    }

}