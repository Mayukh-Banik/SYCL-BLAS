#include <gtest/gtest.h>

#include "syBlast/syBlast.hpp"

TEST(a, a)
{
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    float alpha = 10;
    float* x = sycl::malloc_shared<float>(sizeof(float) * N, q);
    float* y = sycl::malloc_shared<float>(sizeof(float) * N, q);
    int incx = 1;
    int incy = 1;
    syBlast::saxpy(N, alpha, x, incx, y, incy, q);
    q.wait();
    sycl::free(x, q);
    sycl::free(y, q);
}