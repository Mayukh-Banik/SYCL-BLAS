#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>
#include <complex>

/**
 * From i = 0 -> N
 * Y[i * incY] = alpha * X[i * incX] + Y[i * incY]
 * Default would be axpy(N, alpha, X*, 1, Y*, 1) since sycl::queue should select optimal device automatically
 * @details No error checking is done. Done synchronously.
 * @param N The number of indexes to go over
 * @param alpha Value to scale indexes of X
 * @param x Pointer allocated with sycl::malloc*
 * @param incX Increment of X for 1 -> N
 * @param y Pointer allocated with sycl::malloc*. The values in this array gets updated.
 * @param incY Increment of Y for 1 -> N
 */
template <typename A, typename B, typename C>
void axpy(uint64_t N, A alpha, B *x, int incX, C *y, int incY, sycl::queue q = sycl::queue())
{
    q.submit(
         [&](sycl::handler &handler)
         {
             handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                  {
                int64_t Xindex = i[0] * incX;
                int64_t Yindex = i[0] * incY;
                y[Yindex] += (C) alpha * (C) x[Xindex]; });
         })
        .wait();
}

template <typename A, typename B>
void scal(uint64_t N, A alpha, B *x, int incX = 1, sycl::queue q = sycl::queue())
{
    q.submit(
         [&](sycl::handler &handler)
         {
             handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                  {
                int64_t Xindex = i[0] * incX;
                x[Xindex] = (B) alpha * x[Xindex]; });
         })
        .wait();
}

template <typename B, typename C>
void copy(uint64_t N, B *x, int incX, C *y, int incY, sycl::queue q = sycl::queue())
{
    q.submit(
         [&](sycl::handler &handler)
         {
             handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                  {
                int64_t Xindex = i[0] * incX;
                int64_t Yindex = i[0] * incY;
                y[Yindex] = (C) x[Xindex]; });
         })
        .wait();
}

template <typename B, typename C>
void swap(uint64_t N, B *x, int incX, C *y, int incY, sycl::queue q = sycl::queue())
{
    q.submit(
         [&](sycl::handler &handler)
         {
             handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                  {
                int64_t Xindex = i[0] * incX;
                int64_t Yindex = i[0] * incY;
                const C temp = (C) x[Xindex];
                x[Xindex] = (B) y[Yindex];
                y[Yindex] = temp; });
         })
        .wait();
}

template <typename A, typename B, typename C>
A dot(uint64_t N, B *x, int incX, C *y, int incY, sycl::queue q = sycl::queue())
{
    A *dotResult = sycl::malloc_device<A>(sizeof(A), q);
    q.memset(dotResult, 0, sizeof(A)).wait();
    q.submit([&](sycl::handler &cgh)
             {
        auto alpha = sycl::reduction(dotResult, sycl::plus<A>());
        cgh.parallel_for(sycl::range<1>(N), alpha, 
            [=](sycl::id<1> idx, auto& sum) {
                size_t x_idx = idx[0] * incX;
                size_t y_idx = idx[0] * incY;
                sum += static_cast<A>(x[x_idx] * y[y_idx]);
            }); })
        .wait();
    A result;
    q.memcpy(&result, dotResult, sizeof(A)).wait();
    sycl::free(dotResult, q);
    return result;
}

template <typename B, typename A>
B nrm2(uint64_t N, A *x, int incX = 1, sycl::queue q = sycl::queue())
{
    A *dotResult = sycl::malloc_device<A>(sizeof(A) * 1, q);
    q.memset(dotResult, 0, sizeof(A)).wait();
    q.submit([&](sycl::handler &cgh)
             {
        auto alpha = sycl::reduction(dotResult, sycl::plus<A>());
        cgh.parallel_for(sycl::range<1>(N), alpha, 
            [=](sycl::id<1> idx, auto& sum) {
                sum += (x[incX * idx[0]] * x[incX * idx[0]]);
            }); })
        .wait();
    A result;
    q.memcpy(&result, dotResult, sizeof(A)).wait();
    sycl::free(dotResult, q);
    return (B) sycl::sqrt(result);
}