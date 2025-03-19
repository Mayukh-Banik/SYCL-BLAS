#include <SYCL-BLAS/blas.hpp>
#include <gtest/gtest.h>
#include <random>

#define DATA_TYPE float
#define LENGTH 100

class SWAP : public ::testing::Test
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
            y[i] = (DATA_TYPE)(std::rand() % 100);
            y_expected[i] = y[i];
        }
    }

    void TearDown() override
    {
        sycl::free(x, q);
        sycl::free(y, q);
        sycl::free(y_expected, q);
    }
};

TEST_F(SWAP, swap_raw)
{
    swap(LENGTH, x, 1, y, 1, q);
    for (int i = 0; i < LENGTH; i++)
    {
        ASSERT_EQ(x[i], y_expected[i])
            << "x[i]" << x[i] << std::endl;
    }
}

#undef LENGTH
#undef DATA_TYPE

#define DATA_TYPE float
#define LENGTH 100

class SWAPBuffer : public ::testing::Test
{
protected:
    sycl::queue q;
    DATA_TYPE *x_host;
    DATA_TYPE *y_host;
    DATA_TYPE *y_expected;

    void SetUp() override
    {
        std::srand(std::time(nullptr));
        q = sycl::queue();
        x_host = sycl::malloc_host<DATA_TYPE>(LENGTH, q);
        y_host = sycl::malloc_host<DATA_TYPE>(LENGTH, q);
        y_expected = sycl::malloc_host<DATA_TYPE>(LENGTH, q);

        for (int i = 0; i < LENGTH; i++)
        {
            x_host[i] = static_cast<DATA_TYPE>(std::rand() % 100);
            y_host[i] = static_cast<DATA_TYPE>(std::rand() % 100);
            y_expected[i] = y_host[i];
        }
    }

    void TearDown() override
    {
        sycl::free(x_host, q);
        sycl::free(y_host, q);
    }
};

TEST_F(SWAPBuffer, BufferImplementation)
{
    {
        sycl::buffer<DATA_TYPE, 1> x_buf(x_host, sycl::range<1>(LENGTH));
        sycl::buffer<DATA_TYPE, 1> y_buf(y_host, sycl::range<1>(LENGTH));
        swap(LENGTH, x_buf, 1, y_buf, 1, q);
    }
    for (int i = 0; i < LENGTH; i++)
    {
        EXPECT_FLOAT_EQ(x_host[i], y_expected[i])
            << "Mismatch at index " << i
            << ", got: " << y_host[i]
            << ", x[i]: " << x_host[i];
    }
}
