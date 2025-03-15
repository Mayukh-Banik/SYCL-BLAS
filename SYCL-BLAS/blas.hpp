#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

/**
 * From i = 0 -> N
 * Y[i * incY] = alpha * X[i * incX] + Y[i * incY]
 * Default would be axpy(N, alpha, X*, 1, Y*, 1) since sycl::queue should select optimal device automatically
 * @details No error checking is done.
 * @param N The number of indexes to go over
 * @param alpha Value to scale indexes of X
 */
template <typename A, typename B, typename C>
void axpy(uint64_t N, A alpha, B *x, int incX, C *y, int incY, sycl::queue q = sycl::queue())
{
    q.submit(
        [&](sycl::handler &handler)
        {
            handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                 {
                int64_t Xindex = i * incX;
                int64_t Yindex = i * incY;
                y[Yindex] = ((C) alpha * (C) x[Xindex]) + y[Yindex]; });
        });
}

template <typename A, typename B>
void scal(uint64_t N, A alpha, B *x, int incX, sycl::queue q = sycl::queue())
{
    q.submit(
        [&](sycl::handler &handler)
        {
            handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                 {
                int64_t Xindex = i * incX;
                x[Xindex] = ((B) alpha * x[Xindex]); });
        });
}

template <typename A, typename B>
void copy(uint64_t N, A *x, int incX, B *y, int incY, sycl::queue q = sycl::queue())
{
    q.submit(
        [&](sycl::handler &handler)
        {
            handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                 {    
                int64_t Xindex = i * incX;
                int64_t Yindex = i * incY;
                y[Yindex] = (B) x[Xindex]; });
        });
}

template <typename A, typename B>
void swap(uint64_t N, A *x, int incX, B *y, int incY, sycl::queue q = sycl::queue())
{
    q.submit(
        [&](sycl::handler &handler)
        {
            handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                 {    
                int64_t Xindex = i * incX;
                int64_t Yindex = i * incY;
                B temp = y[Yindex];
                y[Yindex] = (B) x[Xindex]; 
                x[Xindex] = (A) y[Yindex]; });
        });
}

template <typename A, typename B, typename C>
C dot(uint64_t N, A *x, int incX, B *y, int incY, sycl::queue q = sycl::queue())
{
    C* val_ptr = sycl::malloc_device<C>(1, q);
    q.memset(val_ptr, 0, sizeof(C)).wait();
    q.submit(
        [&](sycl::handler &handler)
        {
            handler.parallel_for(
                sycl::range<1>(N), [=](sycl::id<1> i) {
                    int64_t Xindex = i * incX;
                    int64_t Yindex = i * incY;
                    C a = (C) x[Xindex] * (C) y[Yindex];
                    auto atomic_ref = sycl::atomic_ref<C, 
                                       sycl::memory_order::relaxed, 
                                       sycl::memory_scope::device, 
                                       sycl::access::address_space::global_space>(*val_ptr);
                    atomic_ref.fetch_add(a);
                });
        }).wait();
    C result;
    q.memcpy(&result, val_ptr, sizeof(C)).wait();
    sycl::free(val_ptr, q);
    return result;
}