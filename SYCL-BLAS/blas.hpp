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

template <typename A, typename B, typename C>
void axpy(uint64_t N, A alpha, sycl::buffer<B, 1> &x, int incX, sycl::buffer<C, 1> &y, int incY, sycl::queue q = sycl::queue())
{
    q.submit([&](sycl::handler &handler)
             {
                auto x_acc = sycl::accessor(x, handler, sycl::read_only);
                auto y_acc = sycl::accessor(y, handler, sycl::read_write);
        handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            int64_t Xindex = i[0] * incX;
            int64_t Yindex = i[0] * incY;
            y_acc[Yindex] += (C) alpha * (C) x_acc[Xindex];
        }); })
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

template <typename A, typename B>
void scal(uint64_t N, A alpha, sycl::buffer<B, 1> &x, int incX = 1, sycl::queue q = sycl::queue())
{
    q.submit([&](sycl::handler &handler)
             {
        auto x_acc = sycl::accessor(x, handler, sycl::read_write);
        handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            int64_t Xindex = i[0] * incX;
            x_acc[Xindex] = static_cast<B>(alpha) * x_acc[Xindex];
        }); })
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
void copy(uint64_t N, sycl::buffer<B, 1> &x, int incX, sycl::buffer<C, 1> &y, int incY, sycl::queue q = sycl::queue())
{
    q.submit([&](sycl::handler &handler)
             {
                auto x_acc = sycl::accessor(x, handler, sycl::read_only);
                auto y_acc = sycl::accessor(y, handler, sycl::write_only);
        handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            int64_t Xindex = i[0] * incX;
            int64_t Yindex = i[0] * incY;
            y_acc[Yindex] = (C) x_acc[Xindex];
        }); })
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

template <typename B, typename C>
void swap(uint64_t N, sycl::buffer<B, 1> &x, int incX, sycl::buffer<C, 1> &y, int incY, sycl::queue q = sycl::queue())
{
    q.submit([&](sycl::handler &handler)
             {
                auto x_acc = sycl::accessor(x, handler, sycl::read_write);
                auto y_acc = sycl::accessor(y, handler, sycl::read_write);
        handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            int64_t Xindex = i[0] * incX;
            int64_t Yindex = i[0] * incY;
            const C temp = (C) x_acc[Xindex];
            x_acc[Xindex] = (B) y_acc[Yindex];
            y_acc[Yindex] = temp;
        }); })
        .wait();
}