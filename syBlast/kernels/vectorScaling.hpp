#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <type_traits>

#if defined(__INTEL_LLVM_COMPILER)
#endif
namespace syBlast::templatedFunctions
{
    // namespace templatedFunctions
    // {

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
//     }
}