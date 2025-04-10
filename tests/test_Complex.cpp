#include "syBlast/core/complex.hpp"

#include <gtest/gtest.h>
#include <complex>
#include <cstdint>
#include <vector>

template <typename T>
bool equalsCC(const std::complex<T> &a, const syBlast::Complex::Complex<T> &b)
{
    if (a.real() == b.real() && a.imag() == b.imag())
    {
        return true;
    }
    return false;
}

template <typename T>
bool equalsCC(const syBlast::Complex::Complex<T> &a, const std::complex<T> &b)
{
    if (a.real() == b.real() && a.imag() == b.imag())
    {
        return true;
    }
    return false;
}

using namespace syBlast::Complex;
using namespace std;

template <typename T>
class ComplexTest : public ::testing::Test
{
};

using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double>;
TYPED_TEST_SUITE(ComplexTest, MyTypes);

TYPED_TEST(ComplexTest, RealInitializaiton)
{
    Complex<TypeParam> a(10);
    complex<TypeParam> b(10);
    EXPECT_TRUE(equalsCC(a, b));
}

TYPED_TEST(ComplexTest, RealImagInitializaiton)
{
    Complex<TypeParam> a(10, 10);
    complex<TypeParam> b(10, 10);
    EXPECT_TRUE(equalsCC(a, b));
}

TYPED_TEST(ComplexTest, RealGet)
{
    Complex<TypeParam> a(10, 10);
    EXPECT_EQ(10, a.real());
}

TYPED_TEST(ComplexTest, ImagGet)
{
    Complex<TypeParam> a(10, 10);
    EXPECT_EQ(10, a.imag());
}

TYPED_TEST(ComplexTest, RealSet)
{
    Complex<TypeParam> a(10, 10);
    a.real(5);
    EXPECT_EQ(5, a.real());
}

TYPED_TEST(ComplexTest, ImagSet)
{
    Complex<TypeParam> a(10, 10);
    a.imag(5);
    EXPECT_EQ(5, a.imag());
}

TYPED_TEST(ComplexTest, UnaryPlus)
{
    Complex<TypeParam> a(3, 4);
    Complex<TypeParam> result = +a;
    EXPECT_TRUE(equalsCC(result, std::complex<TypeParam>(3, 4)));
}

TYPED_TEST(ComplexTest, UnaryMinus)
{
    Complex<TypeParam> a(3, 4);
    Complex<TypeParam> result = -a;
    EXPECT_TRUE(equalsCC(result, std::complex<TypeParam>(-3, -4)));
}

TYPED_TEST(ComplexTest, ComplexAdd)
{
    Complex<TypeParam> a(1, 2), b(3, 4);
    Complex<TypeParam> result = a + b;
    EXPECT_TRUE(equalsCC(result, std::complex<TypeParam>(4, 6)));
}

TYPED_TEST(ComplexTest, ComplexSub)
{
    Complex<TypeParam> a(5, 6), b(2, 1);
    Complex<TypeParam> result = a - b;
    EXPECT_TRUE(equalsCC(result, std::complex<TypeParam>(3, 5)));
}

TYPED_TEST(ComplexTest, ComplexMul)
{
    Complex<TypeParam> a(1, 2), b(3, 4);
    Complex<TypeParam> result = a * b;
    EXPECT_TRUE(equalsCC(result, std::complex<TypeParam>(-5, 10)));
}

TYPED_TEST(ComplexTest, ComplexDiv)
{
    Complex<TypeParam> a(1, 2), b(3, 4);
    Complex<TypeParam> result = a / b;
    std::complex<TypeParam> std_res(1, 2);
    std_res /= std::complex<TypeParam>(3, 4);
    EXPECT_TRUE(equalsCC(result, std_res));
}

TYPED_TEST(ComplexTest, CompoundAdd)
{
    Complex<TypeParam> a(1, 2), b(3, 4);
    a += b;
    EXPECT_TRUE(equalsCC(a, std::complex<TypeParam>(4, 6)));
}

TYPED_TEST(ComplexTest, CompoundSub)
{
    Complex<TypeParam> a(5, 6), b(2, 1);
    a -= b;
    EXPECT_TRUE(equalsCC(a, std::complex<TypeParam>(3, 5)));
}

TYPED_TEST(ComplexTest, CompoundMul)
{
    Complex<TypeParam> a(1, 2), b(3, 4);
    a *= b;
    EXPECT_TRUE(equalsCC(a, std::complex<TypeParam>(-5, 10)));
}

TYPED_TEST(ComplexTest, CompoundDiv)
{
    Complex<TypeParam> a(1, 2), b(3, 4);
    a /= b;
    std::complex<TypeParam> std_res(1, 2);
    std_res /= std::complex<TypeParam>(3, 4);
    EXPECT_TRUE(equalsCC(a, std_res));
}

TYPED_TEST(ComplexTest, ScalarAdd)
{
    Complex<TypeParam> a(1, 2);
    TypeParam s = 3;
    Complex<TypeParam> result = a + s;
    EXPECT_TRUE(equalsCC(result, std::complex<TypeParam>(4, 2)));
}

TYPED_TEST(ComplexTest, ScalarSub)
{
    Complex<TypeParam> a(4, 2);
    TypeParam s = 1;
    Complex<TypeParam> result = a - s;
    EXPECT_TRUE(equalsCC(result, std::complex<TypeParam>(3, 2)));
}

TYPED_TEST(ComplexTest, ScalarMul)
{
    Complex<TypeParam> a(2, 3);
    TypeParam s = 2;
    Complex<TypeParam> result = a * s;
    EXPECT_TRUE(equalsCC(result, std::complex<TypeParam>(4, 6)));
}

TYPED_TEST(ComplexTest, ScalarDiv)
{
    Complex<TypeParam> a(4, 6);
    TypeParam s = 2;
    Complex<TypeParam> result = a / s;
    EXPECT_TRUE(equalsCC(result, std::complex<TypeParam>(2, 3)));
}

TYPED_TEST(ComplexTest, CompoundScalarAdd)
{
    Complex<TypeParam> a(1, 2);
    TypeParam s = 3;
    a += s;
    EXPECT_TRUE(equalsCC(a, std::complex<TypeParam>(4, 2)));
}

TYPED_TEST(ComplexTest, CompoundScalarSub)
{
    Complex<TypeParam> a(4, 2);
    TypeParam s = 1;
    a -= s;
    EXPECT_TRUE(equalsCC(a, std::complex<TypeParam>(3, 2)));
}

TYPED_TEST(ComplexTest, CompoundScalarMul)
{
    Complex<TypeParam> a(2, 3);
    TypeParam s = 2;
    a *= s;
    EXPECT_TRUE(equalsCC(a, std::complex<TypeParam>(4, 6)));
}

TYPED_TEST(ComplexTest, CompoundScalarDiv)
{
    Complex<TypeParam> a(4, 6);
    TypeParam s = 2;
    a /= s;
    EXPECT_TRUE(equalsCC(a, std::complex<TypeParam>(2, 3)));
}

TYPED_TEST(ComplexTest, Equality)
{
    Complex<TypeParam> a(2, 3), b(2, 3);
    EXPECT_TRUE(a == b);
}

TYPED_TEST(ComplexTest, Inequality)
{
    Complex<TypeParam> a(2, 3), b(3, 2);
    EXPECT_TRUE(a != b);
}

TYPED_TEST(ComplexTest, Conjugate)
{
    Complex<TypeParam> a(2, 3);
    Complex<TypeParam> result = a.conjugate();
    EXPECT_TRUE(equalsCC(result, std::complex<TypeParam>(2, -3)));
}

using namespace syBlast::Complex;

class ComplexAdditionTest : public ::testing::Test
{
protected:
    void RunSYCLAddition(std::vector<Complex<float>> &result)
    {
        const size_t N = 1000;

        std::vector<Complex<float>> vec1(N, Complex<float>(1.0, 2.0)); // (1 + 2i)
        std::vector<Complex<float>> vec2(N, Complex<float>(3.0, 4.0)); // (3 + 4i)

        try
        {
            sycl::queue q;
            sycl::buffer<Complex<float>, 1> buf1(vec1.data(), sycl::range<1>(N));
            sycl::buffer<Complex<float>, 1> buf2(vec2.data(), sycl::range<1>(N));
            sycl::buffer<Complex<float>, 1> buf_result(result.data(), sycl::range<1>(N));
            q.submit([&](sycl::handler &h)
                     {
                auto acc1 = buf1.get_access<sycl::access::mode::read>(h);
                auto acc2 = buf2.get_access<sycl::access::mode::read>(h);
                auto acc_result = buf_result.get_access<sycl::access::mode::write>(h);
                h.parallel_for<class vector_addition>(sycl::range<1>(N), [=](sycl::id<1> i) {
                    acc_result[i] = acc1[i] + acc2[i]; 
                }); });
            q.wait();
        }
        catch (const sycl::exception &e)
        {
            std::cerr << "SYCL Exception: " << e.what() << std::endl;
            throw;
        }
    }
};

TEST_F(ComplexAdditionTest, VectorAddition)
{
    const size_t N = 1000;
    std::vector<Complex<float>> result(N);

    RunSYCLAddition(result);

    bool passed = true;
    for (size_t i = 0; i < N; i++)
    {
        Complex<float> expected(4.0, 6.0);
        if (!(result[i] == expected))
        {
            std::cout << "Test failed at index " << i << ": "
                      << "Expected (" << expected.real() << ", " << expected.imag() << ") but got ("
                      << result[i].real() << ", " << result[i].imag() << ")\n";
            passed = false;
        }
    }

    ASSERT_TRUE(passed);
}