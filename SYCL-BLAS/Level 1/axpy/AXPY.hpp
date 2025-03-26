#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <utility>
#include "../../Setup/UsefulFunctions.hpp"

/**
 * @brief Performs the AXPY operation (y = alpha * x + y) on a vector using SYCL.
 *
 * @tparam A The type of the scalar alpha.
 * @tparam B The type of elements in the input vector x (default: A).
 * @tparam C The type of elements in the output vector y (default: A).
 * @param N The number of elements in the vectors.
 * @param alpha The scalar multiplier.
 * @param x Pointer to the input vector.
 * @param incX Stride between consecutive elements in x.
 * @param y Pointer to the output vector.
 * @param incY Stride between consecutive elements in y.
 * @param q The SYCL queue to execute the operation.
 */
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

/**
 * @brief Performs the AXPY operation (y = alpha * x) for complex numbers using SYCL.
 *
 * @tparam A The type of the scalar alpha.
 * @tparam B The type of elements in the input complex vector x (default: A).
 * @tparam C The type of elements in the output complex vector y (default: A).
 * @param N The number of elements in the vectors.
 * @param alpha The scalar multiplier.
 * @param x Pointer to the input complex vector.
 * @param incX Stride between consecutive elements in x.
 * @param y Pointer to the output complex vector.
 * @param incY Stride between consecutive elements in y.
 * @param q The SYCL queue to execute the operation.
 */
template <typename A, typename B = A, typename C = A>
void simpleAxpy(uint64_t N, A alpha, std::complex<B> *x, int incX, std::complex<C> *y, int incY, sycl::queue q)
{
    q.submit([&](sycl::handler &handler)
             { handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                    {
            const int64_t Xindex = i[0] * incX;
            const int64_t Yindex = i[0] * incY;
            y[Yindex] += static_cast<std::complex<C>>(static_cast<std::complex<B>>(alpha) * x[Xindex]); }); });
}

/**
 * @brief Performs the AXPY operation (y = alpha * x) for complex scalar and complex vectors using SYCL.
 *
 * @tparam A The type of the complex scalar alpha.
 * @tparam B The type of elements in the input complex vector x (default: A).
 * @tparam C The type of elements in the output complex vector y (default: A).
 * @param N The number of elements in the vectors.
 * @param alpha The complex scalar multiplier.
 * @param x Pointer to the input complex vector.
 * @param incX Stride between consecutive elements in x.
 * @param y Pointer to the output complex vector.
 * @param incY Stride between consecutive elements in y.
 * @param q The SYCL queue to execute the operation.
 */
template <typename A, typename B = A, typename C = A>
void simpleAxpy(uint64_t N, std::complex<A> alpha, std::complex<B> *x, int incX, std::complex<C> *y, int incY, sycl::queue q)
{
    q.submit([&](sycl::handler &handler)
             { handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                    {
            const int64_t Xindex = i[0] * incX;
            const int64_t Yindex = i[0] * incY;
            y[Yindex] += static_cast<std::complex<C>>(static_cast<std::complex<B>>(alpha) * x[Xindex]); }); });
}
