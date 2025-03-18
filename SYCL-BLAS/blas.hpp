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
 * @param x 
 * @param incX
 * @param y
 * @param incY
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
        }).wait();
}

// template <typename A, typename B>
// void scal(uint64_t N, A alpha, B *x, int incX, sycl::queue q = sycl::queue())
// {
//     q.submit(
//         [&](sycl::handler &handler)
//         {
//             handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
//                                  {
//                 int64_t Xindex = i * incX;
//                 x[Xindex] = ((B) alpha * x[Xindex]); });
//         });
// }

// template <typename A, typename B>
// void copy(uint64_t N, A *x, int incX, B *y, int incY, sycl::queue q = sycl::queue())
// {
//     q.submit(
//         [&](sycl::handler &handler)
//         {
//             handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
//                                  {    
//                 int64_t Xindex = i * incX;
//                 int64_t Yindex = i * incY;
//                 y[Yindex] = (B) x[Xindex]; });
//         });
// }

// template <typename A, typename B>
// void swap(uint64_t N, A *x, int incX, B *y, int incY, sycl::queue q = sycl::queue())
// {
//     q.submit(
//         [&](sycl::handler &handler)
//         {
//             handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
//                                  {    
//                 int64_t Xindex = i * incX;
//                 int64_t Yindex = i * incY;
//                 B temp = y[Yindex];
//                 y[Yindex] = (B) x[Xindex]; 
//                 x[Xindex] = (A) y[Yindex]; });
//         });
// }

// template <typename A, typename B, typename C>
// C dot(uint64_t N, A *x, int incX, B *y, int incY, sycl::queue q = sycl::queue())
// {
//     C *val_ptr = sycl::malloc_device<C>(1, q);
//     q.memset(val_ptr, 0, sizeof(C)).wait();
//     q.submit(
//          [&](sycl::handler &handler)
//          {
//              handler.parallel_for(
//                  sycl::range<1>(N), [=](sycl::id<1> i)
//                  {
//                     int64_t Xindex = i * incX;
//                     int64_t Yindex = i * incY;
//                     C a = (C) x[Xindex] * (C) y[Yindex];
//                     auto atomic_ref = sycl::atomic_ref<C, 
//                                        sycl::memory_order::relaxed, 
//                                        sycl::memory_scope::device, 
//                                        sycl::access::address_space::global_space>(*val_ptr);
//                     atomic_ref.fetch_add(a); });
//          })
//         .wait();
//     C result;
//     q.memcpy(&result, val_ptr, sizeof(C)).wait();
//     sycl::free(val_ptr, q);
//     return result;
// }

// template <typename A, typename B, typename C>
// std::complex<C> dotu(uint64_t N, std::complex<A> *x, int incX, std::complex<B> *y, int incY, sycl::queue q = sycl::queue())
// {
//     std::complex<C> *val_ptr = sycl::malloc_device<std::complex<C>>(1, q);
//     q.memset(val_ptr, 0, sizeof(std::complex<C>)).wait();
//     q.submit([&](sycl::handler &handler)
//              { handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
//                                     {
//             int64_t Xindex = i * incX;
//             int64_t Yindex = i * incY;
//             std::complex<C> a = static_cast<std::complex<C>>(x[Xindex]) * static_cast<std::complex<C>>(y[Yindex]);
//             auto atomic_ref = sycl::atomic_ref<std::complex<C>, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*val_ptr);
//             atomic_ref.fetch_add(a); }); })
//         .wait();
//     std::complex<C> result;
//     q.memcpy(&result, val_ptr, sizeof(std::complex<C>)).wait();
//     sycl::free(val_ptr, q);
//     return result;
// }

// template <typename A, typename B, typename C>
// std::complex<C> dotc(uint64_t N, std::complex<A> *x, int incX, std::complex<B> *y, int incY, sycl::queue q = sycl::queue())
// {
//     std::complex<C> *val_ptr = sycl::malloc_device<std::complex<C>>(1, q);
//     q.memset(val_ptr, 0, sizeof(std::complex<C>)).wait();
//     q.submit([&](sycl::handler &handler)
//              { handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
//                                     {
//             int64_t Xindex = i * incX;
//             int64_t Yindex = i * incY;
//             std::complex<C> a = std::conj(static_cast<std::complex<C>>(x[Xindex])) * static_cast<std::complex<C>>(y[Yindex]);
//             auto atomic_ref = sycl::atomic_ref<std::complex<C>, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*val_ptr);
//             atomic_ref.fetch_add(a); }); })
//         .wait();
//     std::complex<C> result;
//     q.memcpy(&result, val_ptr, sizeof(std::complex<C>)).wait();
//     sycl::free(val_ptr, q);
//     return result;
// }

// template <typename A, typename B>
// B nrm2(uint64_t N, A *x, int incX, sycl::queue q = sycl::queue())
// {
//     B *val_ptr = sycl::malloc_device<C>(1, q);
//     q.memset(val_ptr, 0, sizeof(C)).wait();
//     q.submit(
//          [&](sycl::handler &handler)
//          {
//              handler.parallel_for(
//                  sycl::range<1>(N), [=](sycl::id<1> i)
//                  {
//                     int64_t Xindex = i * incX;
//                     B a = (B) (abs(x[Xindex]) ** 2);
//                     auto atomic_ref = sycl::atomic_ref<C, 
//                                        sycl::memory_order::relaxed, 
//                                        sycl::memory_scope::device, 
//                                        sycl::access::address_space::global_space>(*val_ptr);
//                     atomic_ref.fetch_add(a); });
//          })
//         .wait();
//     B result;
//     q.memcpy(&result, val_ptr, sizeof(B)).wait();
//     sycl::free(val_ptr, q);
//     result = (B)sqrt(result);
//     return result;
// }

// template <typename A, typename C>
// C asum(uint64_t N, std::complex<A> *x, int incX, sycl::queue q = sycl::queue())
// {
//     C *val_ptr = sycl::malloc_device<C>(1, q);
//     q.memset(val_ptr, 0, sizeof(C)).wait();
//     q.submit([&](sycl::handler &handler)
//              { handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
//                                     {
//             int64_t Xindex = i * incX;
//             C a = std::abs(x[Xindex]);
//             auto atomic_ref = sycl::atomic_ref<C, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*val_ptr);
//             atomic_ref.fetch_add(a); }); })
//         .wait();
//     C result;
//     q.memcpy(&result, val_ptr, sizeof(C)).wait();
//     sycl::free(val_ptr, q);
//     return result;
// }

// template <typename A>
// uint64_t i_amax(uint64_t N, std::complex<A> *x, int incX, sycl::queue q = sycl::queue())
// {
//     uint64_t *index_ptr = sycl::malloc_device<uint64_t>(1, q);
//     A *max_val_ptr = sycl::malloc_device<A>(1, q);
//     q.memset(index_ptr, 0, sizeof(uint64_t)).wait();
//     q.memset(max_val_ptr, 0, sizeof(A)).wait();

//     q.submit([&](sycl::handler &handler)
//              { handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
//                                     {
//             int64_t Xindex = i * incX;
//             A abs_val = std::abs(x[Xindex]);
//             if (abs_val > *max_val_ptr) {
//                 *max_val_ptr = abs_val;
//                 *index_ptr = i;
//             } }); })
//         .wait();

//     uint64_t result;
//     q.memcpy(&result, index_ptr, sizeof(uint64_t)).wait();
//     sycl::free(index_ptr, q);
//     sycl::free(max_val_ptr, q);
//     return result;
// }

// template <typename A>
// void rotg(std::complex<A> *a, std::complex<A> *b, A *c, std::complex<A> *s, sycl::queue q = sycl::queue()) {
//     A norm = std::hypot(std::abs(*a), std::abs(*b));
//     *c = std::abs(*a) / norm;
//     *s = (*a) / std::abs(*a) * std::conj(*b) / norm;
//     *a = norm;
//     *b = 0;
// }

// /**
//  * Apply a plane rotation to points (x[i], y[i])
//  * x[i] = c*x[i] + s*y[i]
//  * y[i] = c*y[i] - s*x[i]
//  * @param N The number of elements to process
//  * @param x First vector
//  * @param incX Stride for x
//  * @param y Second vector
//  * @param incY Stride for y
//  * @param c Cosine component of rotation
//  * @param s Sine component of rotation
//  * @param q SYCL queue for execution
//  */
// template <typename T, typename S>
// void rot(uint64_t N, T *x, int incX, T *y, int incY, S c, S s, sycl::queue q = sycl::queue())
// {
//     q.submit(
//         [&](sycl::handler &handler)
//         {
//             handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
//             {
//                 int64_t Xindex = i * incX;
//                 int64_t Yindex = i * incY;
//                 T x_temp = x[Xindex];
//                 T y_temp = y[Yindex];
//                 x[Xindex] = c * x_temp + s * y_temp;
//                 y[Yindex] = c * y_temp - s * x_temp;
//             });
//         }).wait();
// }

// /**
//  * Construct the modified Givens transformation matrix H that zeros
//  * the second component of the vector [sqrt(d1)*x1, sqrt(d2)*y1]^T
//  * @param d1 Input/output scaling factor for x1
//  * @param d2 Input/output scaling factor for y1
//  * @param x1 Input/output x-coordinate 
//  * @param y1 y-coordinate
//  * @param param Output array of size 5 containing parameters for rotm
//  * @param q SYCL queue for execution
//  */
// template <typename T>
// void rotmg(T *d1, T *d2, T *x1, const T y1, T *param, sycl::queue q = sycl::queue())
// {
//     // Execute on host as this is a small calculation
//     q.wait();
    
//     // Flag values for different cases
//     const T ZERO = 0.0;
//     const T ONE = 1.0;
//     const T GAM = 4096.0;
//     const T GAMSQ = GAM * GAM;
//     const T RGAMSQ = 1.0 / GAMSQ;
    
//     T flag, h11, h12, h21, h22;
//     T p1, p2, q1, q2, temp, u;
    
//     if (*d1 < ZERO) {
//         // Set flag for negative case
//         flag = -1.0;
//         h11 = ZERO;
//         h12 = ZERO;
//         h21 = ZERO;
//         h22 = ZERO;
        
//         *d1 = ZERO;
//         *d2 = ZERO;
//         *x1 = ZERO;
//     } else {
//         // Case D1 > 0
//         p2 = *d2 * y1;
//         if (p2 == ZERO) {
//             // Set flag for identity transformation
//             flag = -2.0;
//             param[0] = flag;
//             return;
//         }
        
//         p1 = *d1 * (*x1);
//         q2 = p2 * y1;
//         q1 = p1 * (*x1);
        
//         if (std::abs(q1) > std::abs(q2)) {
//             h21 = -y1 / (*x1);
//             h12 = p2 / p1;
            
//             u = ONE - h12 * h21;
            
//             if (u > ZERO) {
//                 flag = ZERO;
//                 *d1 = *d1 / u;
//                 *d2 = *d2 / u;
//                 *x1 = *x1 * u;
//             }
//         } else {
//             if (q2 < ZERO) {
//                 // Set flag for negative case
//                 flag = -1.0;
//                 h11 = ZERO;
//                 h12 = ZERO;
//                 h21 = ZERO;
//                 h22 = ZERO;
                
//                 *d1 = ZERO;
//                 *d2 = ZERO;
//                 *x1 = ZERO;
//             } else {
//                 flag = 1.0;
//                 h11 = p1 / p2;
//                 h22 = *x1 / y1;
//                 u = ONE + h11 * h22;
//                 temp = *d2 / u;
//                 *d2 = *d1 / u;
//                 *d1 = temp;
//                 *x1 = y1 * u;
//             }
//         }
        
//         // Rescale D1, D2, X1 if necessary
//         if (*d1 != ZERO) {
//             while ((*d1 <= RGAMSQ) || (*d1 >= GAMSQ)) {
//                 if (flag == ZERO) {
//                     h11 = ONE;
//                     h22 = ONE;
//                     flag = -1.0;
//                 } else {
//                     h21 = -h21;
//                     h12 = -h12;
//                     flag = -1.0;
//                 }
                
//                 if (*d1 <= RGAMSQ) {
//                     *d1 = *d1 * GAM * GAM;
//                     *x1 = *x1 / GAM;
//                     h11 = h11 / GAM;
//                     h12 = h12 / GAM;
//                 } else {
//                     *d1 = *d1 / (GAM * GAM);
//                     *x1 = *x1 * GAM;
//                     h11 = h11 * GAM;
//                     h12 = h12 * GAM;
//                 }
//             }
//         }
        
//         if (*d2 != ZERO) {
//             while ((std::abs(*d2) <= RGAMSQ) || (std::abs(*d2) >= GAMSQ)) {
//                 if (flag == ZERO) {
//                     h11 = ONE;
//                     h22 = ONE;
//                     flag = -1.0;
//                 } else {
//                     h21 = -h21;
//                     h12 = -h12;
//                     flag = -1.0;
//                 }
                
//                 if (std::abs(*d2) <= RGAMSQ) {
//                     *d2 = *d2 * GAM * GAM;
//                     h21 = h21 / GAM;
//                     h22 = h22 / GAM;
//                 } else {
//                     *d2 = *d2 / (GAM * GAM);
//                     h21 = h21 * GAM;
//                     h22 = h22 * GAM;
//                 }
//             }
//         }
//     }
    
//     // Store flag and h parameters
//     param[0] = flag;
//     if (flag < ZERO) {
//         // For flag = -1.0, set H to identity matrix
//         if (flag == -1.0) {
//             param[1] = h11;
//             param[2] = h21;
//             param[3] = h12;
//             param[4] = h22;
//         }
//     } else {
//         // For flag = 0.0, set H based on h11, h21, h12, h22
//         if (flag == ZERO) {
//             param[1] = h11;
//             param[2] = h21;
//             param[3] = h12;
//             param[4] = h22;
//         } else {
//             // For flag = 1.0, set H based on h11, h22
//             param[1] = h11;
//             param[2] = h21;
//             param[3] = h12;
//             param[4] = h22;
//         }
//     }
// }

// /**
//  * Apply the modified Givens rotation to the vectors x and y
//  * @param N The number of elements to process
//  * @param x First vector
//  * @param incX Stride for x
//  * @param y Second vector
//  * @param incY Stride for y
//  * @param param Array of size 5 containing parameters from rotmg
//  * @param q SYCL queue for execution
//  */
// template <typename T>
// void rotm(uint64_t N, T *x, int incX, T *y, int incY, T *param, sycl::queue q = sycl::queue())
// {
//     T flag = param[0];
    
//     // Early return for identity transformation
//     if (flag == -2.0) {
//         return;
//     }
    
//     q.submit(
//         [&](sycl::handler &handler)
//         {
//             handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
//             {
//                 int64_t Xindex = i * incX;
//                 int64_t Yindex = i * incY;
//                 T x_temp = x[Xindex];
//                 T y_temp = y[Yindex];
                
//                 if (flag < 0.0) {
//                     // H is a unit matrix
//                     if (flag == -1.0) {
//                         x[Xindex] = param[1] * x_temp + param[3] * y_temp;
//                         y[Yindex] = param[2] * x_temp + param[4] * y_temp;
//                     }
//                 } else if (flag == 0.0) {
//                     // H has the form:
//                     // [  1      param[3] ]
//                     // [ param[2]    1    ]
//                     x[Xindex] = x_temp + param[3] * y_temp;
//                     y[Yindex] = param[2] * x_temp + y_temp;
//                 } else {
//                     // H has the form:
//                     // [ param[1]  param[3] ]
//                     // [ param[2]  param[4] ]
//                     x[Xindex] = param[1] * x_temp + param[3] * y_temp;
//                     y[Yindex] = param[2] * x_temp + param[4] * y_temp;
//                 }
//             });
//         }).wait();
// }