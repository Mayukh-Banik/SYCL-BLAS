#include <SYCL-BLAS/blas.hpp>
#include <gtest/gtest.h>
#include <random>

#define DATA_TYPE float
#define LENGTH 100

class COPY : public ::testing::Test
{
protected:
    sycl::queue q;
    DATA_TYPE *x;
    DATA_TYPE *y;
    DATA_TYPE *y_expected;

    void SetUp() override
    {
        std::srand(std::time(nullptr));
        q = sycl::queue();
        x = sycl::malloc_shared<DATA_TYPE>(sizeof(DATA_TYPE) * LENGTH, q);
        y = sycl::malloc_shared<DATA_TYPE>(sizeof(DATA_TYPE) * LENGTH, q);
        y_expected = sycl::malloc_shared<DATA_TYPE>(sizeof(DATA_TYPE) * LENGTH, q);

        for (int i = 0; i < LENGTH; i++)
        {
            x[i] = (DATA_TYPE)(std::rand() % 100);
            // y[i] = (DATA_TYPE)(std::rand() % 100);
            // y_expected[i] = y[i];
        }
    }

    void TearDown() override
    {
        sycl::free(x, q);
        sycl::free(y, q);
    }
};

TEST_F(COPY, copy_raw)
{
    copy(LENGTH, x, 1, y, 1, q);
    for (int i = 0; i < LENGTH; i++)
    {
        ASSERT_EQ(x[i], y[i])
            << "x[i]" << x[i] << std::endl;
    }
}

#undef LENGTH
#undef DATA_TYPE
