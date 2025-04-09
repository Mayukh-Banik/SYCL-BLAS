#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <complex>

namespace syBlast
{
    namespace templatedFunctions
    {

        template <typename A, typename B = A, typename C = A>
        void axpy(
            const uint64_t N,
            const A alpha,
            const B *x,
            const int incx,
            C *y,
            const int incy,
            sycl::queue &q)
        {

            q.submit(
                [&](sycl::handler &handler)
                {
                    handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                         {
                   const int64_t Xindex = i[0] * incx;
                   const int64_t Yindex = i[0] * incy;
                   
                   if constexpr (std::is_same_v<A, B> && std::is_same_v<B, C>)
                   {
                    y[Yindex] += alpha * x[Xindex]; 
                   }
                   else
                   {
                    y[Yindex] += (C) alpha * (C) x[Xindex]; 
                   } });
                });
        }

        template <typename A, typename B = A>
        void scal(
            const uint64_t N,
            const A alpha,
            B *x,
            const int incx,
            sycl::queue &q)
        {
            q.submit(
                [&](sycl::handler &handler)
                {
                    handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                         {
                   const int64_t Xindex = i[0] * incx;
                   
                   if constexpr (std::is_same_v<A, B>)
                   {
                    x[Xindex] = alpha * x[Xindex]; 

                   }
                   else
                   {
                    x[Xindex] = ((B) alpha) * x[Xindex]; 
                   } });
                });
        }

        template <typename A, typename B = A>
        void copy(
            const uint64_t N,
            const A *x,
            const int incx,
            B *y,
            const int incy,
            sycl::queue &q)
        {
            q.submit(
                [&](sycl::handler &handler)
                {
                    handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                         {
               const int64_t Xindex = i[0] * incx;
               const int64_t Yindex = i[0] * incy;
               
               if constexpr (std::is_same_v<A, B>)
               {
                y[Yindex] = x[Xindex]; 
               }
               else
               {
                    y[Yindex] = (B) x[Xindex]; 
               } });
                });
        }

        template <typename A, typename B = A>
        void swap(
            const uint64_t N,
            A *x,
            const int incx,
            B *y,
            const int incy,
            sycl::queue &q)
        {
            q.submit(
                [&](sycl::handler &handler)
                {
                    handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                         {
               const int64_t Xindex = i[0] * incx;
               const int64_t Yindex = i[0] * incy;
               
               if constexpr (std::is_same_v<A, B>)
               {
                   auto temp = x[Xindex];
                   x[Xindex] = y[Yindex];
                   y[Yindex] = temp;
               }
               else
               {
                   auto temp = x[Xindex];
                   x[Xindex] = static_cast<A>(y[Yindex]);
                   y[Yindex] = static_cast<B>(temp);
               } });
                });
        }

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
        std::complex<C> dotc(
            const uint64_t N,
            const std::complex<A> *x,
            const int incx,
            const std::complex<B> *y,
            const int incy,
            sycl::queue &q)
        {
            using ResultT = std::complex<C>;
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
