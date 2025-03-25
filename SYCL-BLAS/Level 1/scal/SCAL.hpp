#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <utility>
#include "../../Setup/UsefulFunctions.hpp"
#include <complex>

/**
 * @brief Scales a vector by a scalar value using SYCL.
 *
 * @tparam A The type of the scalar alpha.
 * @tparam B The type of elements in the vector x (default: A).
 * @param N The number of elements in the vector.
 * @param alpha The scalar multiplier.
 * @param x Pointer to the input/output vector.
 * @param incX Stride between consecutive elements in x.
 * @param q The SYCL queue to execute the operation (default: new SYCL queue).
 * @param Async Whether to execute asynchronously (currently unused).
 * @param index Optional index parameter (default: -1, currently unused).
 */
template <typename A, typename B = A>
void scal(uint64_t N, A alpha, B* x, int incX, sycl::queue q = sycl::queue(), bool Async = false, int index = -1)
{
    q.submit(
        [&](sycl::handler& handler)
        {
            handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
            {
                const int64_t index = i[0] * incX;
                x[index] = (B) alpha * x[index];
            });
        }
    );
}

/**
 * @brief Scales a complex vector by a scalar value using SYCL.
 *
 * @tparam A The type of the scalar alpha.
 * @tparam B The type of elements in the complex vector x (default: A).
 * @param N The number of elements in the vector.
 * @param alpha The scalar multiplier.
 * @param x Pointer to the input/output complex vector.
 * @param incX Stride between consecutive elements in x.
 * @param q The SYCL queue to execute the operation (default: new SYCL queue).
 * @param Async Whether to execute asynchronously (currently unused).
 * @param index Optional index parameter (default: -1, currently unused).
 */
template <typename A, typename B = A>
void scal(uint64_t N, A alpha, std::complex<B>* x, int incX, sycl::queue q = sycl::queue(), bool Async = false, int index = -1)
{
    q.submit(
        [&](sycl::handler& handler)
        {
            handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
            {
                const int64_t index = i[0] * incX;
                x[index] = static_cast<std::complex<B>>(alpha) * x[index];
            });
        }
    );
}
