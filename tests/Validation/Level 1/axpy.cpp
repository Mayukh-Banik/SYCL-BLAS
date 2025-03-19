#include <SYCL-BLAS/blas.hpp>
#include <gtest/gtest.h>
#include <random>

#define DATA_TYPE float
#define LENGTH 100

class AXPY : public ::testing::Test
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
            x[i] = (DATA_TYPE) (std::rand() % 100);
            y[i] = (DATA_TYPE) (std::rand() % 100);
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

TEST_F(AXPY, axpy_raw)
{
    int alpha = 1;
    axpy(LENGTH, alpha, x, 1, y, 1, q);
    for (int i = 0; i < LENGTH; i++)
    {
        ASSERT_EQ(y_expected[i] + alpha * x[i], y[i])
        << "x[i]" << x[i] << std::endl;
    }
}

#undef LENGTH
#undef DATA_TYPE

#define DATA_TYPE float
#define LENGTH 100

class AXPYBuffer : public ::testing::Test
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
        sycl::free(y_expected, q);
    }
};

TEST_F(AXPYBuffer, BufferImplementation)
{
    DATA_TYPE alpha = 1;
    
    // Create buffers from the host data
    {
        // Using scope to ensure buffers are destroyed and data is synchronized back
        sycl::buffer<DATA_TYPE, 1> x_buf(x_host, sycl::range<1>(LENGTH));
        sycl::buffer<DATA_TYPE, 1> y_buf(y_host, sycl::range<1>(LENGTH));
        
        // Call the buffer version of axpy
        axpy(LENGTH, alpha, x_buf, 1, y_buf, 1, q);
        
        // Buffers go out of scope here, which triggers synchronization back to host
    }
    
    // Verify results
    for (int i = 0; i < LENGTH; i++)
    {
        EXPECT_FLOAT_EQ(y_expected[i] + alpha * x_host[i], y_host[i])
            << "Mismatch at index " << i 
            << ", expected: " << (y_expected[i] + alpha * x_host[i])
            << ", got: " << y_host[i]
            << ", x[i]: " << x_host[i]
            << ", original y[i]: " << y_expected[i];
    }
}
