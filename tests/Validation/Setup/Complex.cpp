// #include <gtest/gtest.h>
// // #include <complex>
// #include "SYCL-BLAS/Setup/Complex.hpp"
// // #include <cstdlib>
// #include <ctime>

// template <typename T>
// class ComplexTest : public ::testing::Test
// {
// protected:
//     void SetUp() override
//     {
//         std::srand(std::time(0));
//         randomNum1 = std::rand() % 100;
//         randomNum2 = std::rand()% 100;
//         randomNum3 = std::rand()% 100;
//         randomNum4 = std::rand()% 100;
//         a = syBlast::Complex<T>(randomNum1, randomNum2);
//         b = syBlast::Complex<T>(randomNum3, randomNum4);
//         std_a = std::complex<T>(randomNum1, randomNum2);
//         std_b = std::complex<T>(randomNum3, randomNum4);
//     }

//     int randomNum1, randomNum2, randomNum3, randomNum4;
//     std::complex<T> std_a, std_b;
//     syBlast::Complex<T> a, b;
// };

// using TestTypes = ::testing::Types<int, float, double, long double>;
// TYPED_TEST_SUITE(ComplexTest, TestTypes);

// TYPED_TEST(ComplexTest, Addition)
// {
//     sqrt()
//     auto c = this->a + this->b;
//     auto d = this->std_a + this->std_b;
//     EXPECT_EQ(c, d) << "a: " << this->a << ", b: " << this->b << ", std_a: " << this->std_a << ", std_b: " << this->std_b
//                     << ", randomNum1: " << this->randomNum1
//                     << ", randomNum2: " << this->randomNum2
//                     << ", randomNum3: " << this->randomNum3
//                     << ", randomNum4: " << this->randomNum4;
// }

// TYPED_TEST(ComplexTest, Subtraction)
// {
//     auto c = this->a - this->b;
//     auto d = this->std_a - this->std_b;
//     EXPECT_EQ(c, d) << "a: " << this->a << ", b: " << this->b << ", std_a: " << this->std_a << ", std_b: " << this->std_b
//                     << ", randomNum1: " << this->randomNum1
//                     << ", randomNum2: " << this->randomNum2
//                     << ", randomNum3: " << this->randomNum3
//                     << ", randomNum4: " << this->randomNum4;
// }

// TYPED_TEST(ComplexTest, Multiplication)
// {
//     auto c = this->a * this->b;
//     auto d = this->std_a * this->std_b;
//     EXPECT_EQ(c, d) << "a: " << this->a << ", b: " << this->b << ", std_a: " << this->std_a << ", std_b: " << this->std_b
//                     << ", randomNum1: " << this->randomNum1
//                     << ", randomNum2: " << this->randomNum2
//                     << ", randomNum3: " << this->randomNum3
//                     << ", randomNum4: " << this->randomNum4;
// }

// TYPED_TEST(ComplexTest, Division)
// {
//     auto c = this->a / this->b;
//     auto d = this->std_a / this->std_b;
//     EXPECT_EQ(c, d) << "a: " << this->a << ", b: " << this->b << ", std_a: " << this->std_a << ", std_b: " << this->std_b
//                     << ", randomNum1: " << this->randomNum1
//                     << ", randomNum2: " << this->randomNum2
//                     << ", randomNum3: " << this->randomNum3
//                     << ", randomNum4: " << this->randomNum4;
// }

// TYPED_TEST(ComplexTest, AdditionAssignment)
// {
//     auto c = this->a;
//     c += this->b;
//     auto d = this->std_a;
//     d += this->std_b;

//     EXPECT_EQ(c, d) << "a: " << this->a << ", b: " << this->b << ", std_a: " << this->std_a << ", std_b: " << this->std_b
//                     << ", randomNum1: " << this->randomNum1
//                     << ", randomNum2: " << this->randomNum2
//                     << ", randomNum3: " << this->randomNum3
//                     << ", randomNum4: " << this->randomNum4;
// }

// TYPED_TEST(ComplexTest, SubtractionAssignment)
// {
//     auto c = this->a;
//     c -= this->b;
//     auto d = this->std_a;
//     d -= this->std_b;

//     EXPECT_EQ(c, d) << "a: " << this->a << ", b: " << this->b << ", std_a: " << this->std_a << ", std_b: " << this->std_b
//                     << ", randomNum1: " << this->randomNum1
//                     << ", randomNum2: " << this->randomNum2
//                     << ", randomNum3: " << this->randomNum3
//                     << ", randomNum4: " << this->randomNum4;
// }

// TYPED_TEST(ComplexTest, MultiplicationAssignment)
// {
//     auto c = this->a;
//     c *= this->b;
//     auto d = this->std_a;
//     d *= this->std_b;

//     EXPECT_EQ(c, d) << "a: " << this->a << ", b: " << this->b << ", std_a: " << this->std_a << ", std_b: " << this->std_b
//                     << ", randomNum1: " << this->randomNum1
//                     << ", randomNum2: " << this->randomNum2
//                     << ", randomNum3: " << this->randomNum3
//                     << ", randomNum4: " << this->randomNum4;
// }

// TYPED_TEST(ComplexTest, DivisionAssignment)
// {
//     auto c = this->a;
//     EXPECT_NO_THROW(c /= this->b);
//     auto d = this->std_a;
//     EXPECT_NO_THROW(d /= this->std_b);
//     EXPECT_EQ(c, d) << "a: " << this->a << ", b: " << this->b << ", std_a: " << this->std_a << ", std_b: " << this->std_b
//                     << ", randomNum1: " << this->randomNum1
//                     << ", randomNum2: " << this->randomNum2
//                     << ", randomNum3: " << this->randomNum3
//                     << ", randomNum4: " << this->randomNum4;

// }

// TYPED_TEST(ComplexTest, RealPart)
// {
//     EXPECT_EQ(syBlast::real(this->a), std::real(this->std_a))
//         << "a: " << this->a << ", std_a: " << this->std_a;
// }

// TYPED_TEST(ComplexTest, ImaginaryPart)
// {
//     EXPECT_EQ(syBlast::imag(this->a), std::imag(this->std_a))
//         << "a: " << this->a << ", std_a: " << this->std_a;
// }

// TYPED_TEST(ComplexTest, AbsoluteValue)
// {
//     EXPECT_EQ(syBlast::abs(this->a), std::abs(this->std_a))
//         << "a: " << this->a << ", std_a: " << this->std_a;
// }

// // TYPED_TEST(ComplexTest, Argument)
// // {
// //     EXPECT_DOUBLE_EQ(syBlast::arg(this->a), std::arg(this->std_a))
// //         << "a: " << this->a << ", std_a: " << this->std_a;
// // }

// // TYPED_TEST(ComplexTest, Norm)
// // {
// //     EXPECT_DOUBLE_EQ(syBlast::norm(this->a), std::norm(this->std_a))
// //         << "a: " << this->a << ", std_a: " << this->std_a;
// // }

// // TYPED_TEST(ComplexTest, Conjugate)
// // {
// //     auto c = syBlast::conj(this->a);
// //     auto d = std::conj(this->std_a);
// //     EXPECT_EQ(c, d) << "a: " << this->a << ", std_a: " << this->std_a;
// // }

#include <gtest/gtest.h>

TEST(a, a)
{
    
}