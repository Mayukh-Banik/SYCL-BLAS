#include <SYCL-BLAS/blas.hpp>
#include <gtest/gtest.h>

int N = 100;
#define DATA_TYPE float
TEST(nrm2, basic)
{
    sycl::queue q = sycl::queue();
    auto x = sycl::malloc_shared<DATA_TYPE>(sizeof(DATA_TYPE) * N, q);

    for (int i = 0; i < N; i++)
    {
        x[i] = 1;
    }

    auto y = nrm2<int>(N, x, 1, q);
    sycl::free(x, q);
    ASSERT_FLOAT_EQ(sycl::sqrt(100.0f), y);

}