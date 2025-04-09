#include <gtest/gtest.h>

#include "syBlast/syBlast.hpp"
#include <random>
#include <algorithm>

template <typename T>
class testAXPY
{
public:
    sycl::queue q;
    uint64_t N;
    T *x;
    T *y;
    T *expY;
    T alpha;

    testAXPY(uint64_t N, T alpha, sycl::queue &q, int seed)
    {
        this->q = q;
        this->N = N;

        this->x = sycl::malloc_shared<T>(sizeof(T) * N, q);
        this->y = sycl::malloc_shared<T>(sizeof(T) * N, q);
        this->expY = sycl::malloc_shared<T>(sizeof(T) * N, q);

        std::mt19937 gen(seed);

        std::uniform_int_distribution<int> dist(-100, 100);

        this->alpha = alpha;

        for (uint64_t i = 0; i < N; i++) 
        {
            this->x[i] = static_cast<T>(dist(gen));
            this->y[i] = static_cast<T>(dist(gen));
            this->expY[i] = this->y[i] + this->alpha * this->x[i];
        }
    }

    ~testAXPY()
    {
        sycl::free(this->x, this->q);
        sycl::free(this->y, this->q);
        sycl::free(this->expY, this->q);
    }
};

TEST(AXPY, SAXPY1)
{
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    float alpha = 10;
    testAXPY<float> tempclass(N, alpha, q, 1);
    syBlast::saxpy(N, alpha, tempclass.x, 1, tempclass.y, 1, q);
    q.wait();
    for (uint64_t i = 0; i < tempclass.N; i++)
    {
        ASSERT_FLOAT_EQ(tempclass.y[i], tempclass.expY[i])
        << " Alpha: " << tempclass.alpha
        << " X[i]: " << tempclass.x[i]
        << " Y[i]: " << tempclass.y[i]
        << " Expy[i]: " << tempclass.expY[i];
    }
}

TEST(AXPY, DAXPY1)
{
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    double alpha = 10;
    testAXPY<double> tempclass(N, alpha, q, 1);
    syBlast::daxpy(N, alpha, tempclass.x, 1, tempclass.y, 1, q);
    q.wait();
    for (uint64_t i = 0; i < tempclass.N; i++)
    {
        ASSERT_FLOAT_EQ(tempclass.y[i], tempclass.expY[i])
        << " Alpha: " << tempclass.alpha
        << " X[i]: " << tempclass.x[i]
        << " Y[i]: " << tempclass.y[i]
        << " Expy[i]: " << tempclass.expY[i];
    }
}

TEST(AXPY, CAXPY1)
{
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    std::complex<float> alpha = 10;
    testAXPY<std::complex<float>> tempclass(N, alpha, q, 1);
    syBlast::caxpy(N, alpha, tempclass.x, 1, tempclass.y, 1, q);
    q.wait();
    for (uint64_t i = 0; i < tempclass.N; i++)
    {
        ASSERT_EQ(tempclass.y[i], tempclass.expY[i])
        << " Alpha: " << tempclass.alpha
        << " X[i]: " << tempclass.x[i]
        << " Y[i]: " << tempclass.y[i]
        << " Expy[i]: " << tempclass.expY[i];
    }
}

TEST(AXPY, ZAXPY1)
{
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    std::complex<double> alpha = 10;
    testAXPY<std::complex<double>> tempclass(N, alpha, q, 1);
    syBlast::zaxpy(N, alpha, tempclass.x, 1, tempclass.y, 1, q);
    q.wait();
    for (uint64_t i = 0; i < tempclass.N; i++)
    {
        ASSERT_EQ(tempclass.y[i], tempclass.expY[i])
        << " Alpha: " << tempclass.alpha
        << " X[i]: " << tempclass.x[i]
        << " Y[i]: " << tempclass.y[i]
        << " Expy[i]: " << tempclass.expY[i];
    }
}