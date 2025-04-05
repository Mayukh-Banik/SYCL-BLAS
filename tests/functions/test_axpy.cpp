#include "syBlast/blas.hpp"
#include <gtest/gtest.h>
#include <type_traits>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <sycl/sycl.hpp>

template <typename T>
class AxpyTester
{
public:
    sycl::queue q;
    T *a;
    T *b;
    T *c;
    T alpha;
    uint64_t N;

    AxpyTester(uint64_t N, T alpha, int seed = 0)
        : q(sycl::queue()), N(N), alpha(alpha)
    {
        a = sycl::malloc_shared<T>(N, q);
        b = sycl::malloc_shared<T>(N, q);
        c = sycl::malloc_shared<T>(N, q);

        std::srand(seed);
        for (uint64_t i = 0; i < N; ++i)
        {
            a[i] = static_cast<T>(std::rand()) / static_cast<T>(INT16_MAX);
            b[i] = static_cast<T>(std::rand()) / static_cast<T>(INT16_MAX);
            c[i] = a[i] * alpha + b[i];
        }
    }

    void axpyFunc()
    {
        if constexpr (std::is_same_v<T, float>)
        {
            syBlast::saxpy(N, alpha, a, 1, b, 1, q);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            syBlast::daxpy(N, alpha, a, 1, b, 1, q);
        }
        else if constexpr (std::is_same_v<T, std::complex<float>>)
        {
            syBlast::caxpy(N, alpha, a, 1, b, 1, q);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>)
        {
            syBlast::zaxpy(N, alpha, a, 1, b, 1, q);
        }
        else
        {
            static_assert(!std::is_same_v<T, T>, "Unsupported type for axpyFunc");
        }
        q.wait();
    }

    bool isEqual()
    {
        for (uint64_t i = 0; i < N; i++)
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                if (std::abs(b[i] - c[i]) > 1e-5)
                    return false;
            }
            else if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
            {
                if (std::abs(b[i] - c[i]) > 1e-5)
                    return false;
            }
        }
        return true;
    }

    ~AxpyTester()
    {
        sycl::free(a, q);
        sycl::free(b, q);
        sycl::free(c, q);
    }
};

// --- TYPED TESTS SETUP ---

template <typename T>
class TypedAxpyTest : public ::testing::Test {};

using MyTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(TypedAxpyTest, MyTypes);

TYPED_TEST(TypedAxpyTest, AxpyRunsAndValidates)
{
    AxpyTester<TypeParam> temp(100, TypeParam(1));
    temp.axpyFunc();
    ASSERT_TRUE(temp.isEqual());
}
