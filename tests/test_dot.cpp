#include <gtest/gtest.h>
#include "syBlast/syBlast.hpp"
#include <random>
#include <algorithm>
#include <complex>

// Test fixture for standard dot products (sdot, ddot)
template <typename T>
class testDOT {
public:
    sycl::queue q;
    uint64_t N;
    T *x;
    T *y;
    T expectedResult;

    testDOT(uint64_t N, sycl::queue &q, int seed) {
        this->q = q;
        this->N = N;
        this->x = sycl::malloc_shared<T>(sizeof(T) * N, q);
        this->y = sycl::malloc_shared<T>(sizeof(T) * N, q);
        
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(-100, 100);
        
        this->expectedResult = 0;
        for (uint64_t i = 0; i < N; i++) {
            this->x[i] = static_cast<T>(dist(gen));
            this->y[i] = static_cast<T>(dist(gen));
            this->expectedResult += this->x[i] * this->y[i];
        }
    }

    ~testDOT() {
        sycl::free(this->x, this->q);
        sycl::free(this->y, this->q);
    }
};

// Test fixture for complex dot products (cdotu, zdotu, cdotc, zdotc)
template <typename T>
class testComplexDOT {
public:
    sycl::queue q;
    uint64_t N;
    std::complex<T> *x;
    std::complex<T> *y;
    std::complex<T> expectedDotu;
    std::complex<T> expectedDotc;

    testComplexDOT(uint64_t N, sycl::queue &q, int seed) {
        this->q = q;
        this->N = N;
        this->x = sycl::malloc_shared<std::complex<T>>(sizeof(std::complex<T>) * N, q);
        this->y = sycl::malloc_shared<std::complex<T>>(sizeof(std::complex<T>) * N, q);
        
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(-100, 100);
        
        this->expectedDotu = std::complex<T>(0, 0);
        this->expectedDotc = std::complex<T>(0, 0);
        
        for (uint64_t i = 0; i < N; i++) {
            T real_x = static_cast<T>(dist(gen));
            T imag_x = static_cast<T>(dist(gen));
            T real_y = static_cast<T>(dist(gen));
            T imag_y = static_cast<T>(dist(gen));
            
            this->x[i] = std::complex<T>(real_x, imag_x);
            this->y[i] = std::complex<T>(real_y, imag_y);
            
            // Compute expected dotu: x[i] * y[i]
            this->expectedDotu += this->x[i] * this->y[i];
            
            // Compute expected dotc: conj(x[i]) * y[i]
            this->expectedDotc += std::conj(this->x[i]) * this->y[i];
        }
    }

    ~testComplexDOT() {
        sycl::free(this->x, this->q);
        sycl::free(this->y, this->q);
    }
};

// Test fixture for mixed precision dot products (sdsdot, dsdot)
class testMixedDOT {
public:
    sycl::queue q;
    uint64_t N;
    float *x_float;
    double *x_double;
    float *y;
    float alpha;
    float expectedSdsdot;
    float expectedDsdot;

    testMixedDOT(uint64_t N, float alpha, sycl::queue &q, int seed) {
        this->q = q;
        this->N = N;
        this->alpha = alpha;
        this->x_float = sycl::malloc_shared<float>(sizeof(float) * N, q);
        this->x_double = sycl::malloc_shared<double>(sizeof(double) * N, q);
        this->y = sycl::malloc_shared<float>(sizeof(float) * N, q);
        
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(-100, 100);
        
        float dotProduct = 0.0f;
        double dsdotProduct = 0.0;
        
        for (uint64_t i = 0; i < N; i++) {
            float x_val = static_cast<float>(dist(gen));
            this->x_float[i] = x_val;
            this->x_double[i] = static_cast<double>(x_val);
            this->y[i] = static_cast<float>(dist(gen));
            
            dotProduct += this->x_float[i] * this->y[i];
            dsdotProduct += static_cast<double>(this->x_double[i]) * static_cast<double>(this->y[i]);
        }
        
        // For sdsdot: result = alpha + sum(x[i] * y[i])
        this->expectedSdsdot = alpha + dotProduct;
        
        // For dsdot: result = sum(double(x[i]) * double(y[i])) as float
        this->expectedDsdot = static_cast<float>(dsdotProduct);
    }

    ~testMixedDOT() {
        sycl::free(this->x_float, this->q);
        sycl::free(this->x_double, this->q);
        sycl::free(this->y, this->q);
    }
};

// Tests for sdot
TEST(DOT, SDOT1) {
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    testDOT<float> tempclass(N, q, 42);
    
    float result = syBlast::sdot(tempclass.N, tempclass.x, 1, tempclass.y, 1, q);
    q.wait();
    
    ASSERT_FLOAT_EQ(result, tempclass.expectedResult)
        << "Expected: " << tempclass.expectedResult
        << " Got: " << result;
}

// Tests for ddot
TEST(DOT, DDOT1) {
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    testDOT<double> tempclass(N, q, 42);
    
    double result = syBlast::ddot(tempclass.N, tempclass.x, 1, tempclass.y, 1, q);
    q.wait();
    
    ASSERT_DOUBLE_EQ(result, tempclass.expectedResult)
        << "Expected: " << tempclass.expectedResult
        << " Got: " << result;
}

// Tests for cdotu
TEST(DOT, CDOTU1) {
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    testComplexDOT<float> tempclass(N, q, 42);
    
    std::complex<float> result = syBlast::cdotu(tempclass.N, tempclass.x, 1, tempclass.y, 1, q);
    q.wait();
    
    ASSERT_FLOAT_EQ(result.real(), tempclass.expectedDotu.real())
        << "Real part - Expected: " << tempclass.expectedDotu.real()
        << " Got: " << result.real();
    ASSERT_FLOAT_EQ(result.imag(), tempclass.expectedDotu.imag())
        << "Imaginary part - Expected: " << tempclass.expectedDotu.imag()
        << " Got: " << result.imag();
}

// Tests for zdotu
TEST(DOT, ZDOTU1) {
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    testComplexDOT<double> tempclass(N, q, 42);
    
    std::complex<double> result = syBlast::zdotu(tempclass.N, tempclass.x, 1, tempclass.y, 1, q);
    q.wait();
    
    ASSERT_DOUBLE_EQ(result.real(), tempclass.expectedDotu.real())
        << "Real part - Expected: " << tempclass.expectedDotu.real()
        << " Got: " << result.real();
    ASSERT_DOUBLE_EQ(result.imag(), tempclass.expectedDotu.imag())
        << "Imaginary part - Expected: " << tempclass.expectedDotu.imag()
        << " Got: " << result.imag();
}

// Tests for cdotc
TEST(DOT, CDOTC1) {
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    testComplexDOT<float> tempclass(N, q, 42);
    
    std::complex<float> result = syBlast::cdotc(tempclass.N, tempclass.x, 1, tempclass.y, 1, q);
    q.wait();
    
    ASSERT_FLOAT_EQ(result.real(), tempclass.expectedDotc.real())
        << "Real part - Expected: " << tempclass.expectedDotc.real()
        << " Got: " << result.real();
    ASSERT_FLOAT_EQ(result.imag(), tempclass.expectedDotc.imag())
        << "Imaginary part - Expected: " << tempclass.expectedDotc.imag()
        << " Got: " << result.imag();
}

// Tests for zdotc
TEST(DOT, ZDOTC1) {
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    testComplexDOT<double> tempclass(N, q, 42);
    
    std::complex<double> result = syBlast::zdotc(tempclass.N, tempclass.x, 1, tempclass.y, 1, q);
    q.wait();
    
    ASSERT_DOUBLE_EQ(result.real(), tempclass.expectedDotc.real())
        << "Real part - Expected: " << tempclass.expectedDotc.real()
        << " Got: " << result.real();
    ASSERT_DOUBLE_EQ(result.imag(), tempclass.expectedDotc.imag())
        << "Imaginary part - Expected: " << tempclass.expectedDotc.imag()
        << " Got: " << result.imag();
}

// Tests for sdsdot
TEST(DOT, SDSDOT1) {
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    float alpha = 5.0f;
    testMixedDOT tempclass(N, alpha, q, 42);
    
    float result = syBlast::sdsdot(tempclass.N, alpha, tempclass.x_float, 1, tempclass.y, 1, q);
    q.wait();
    
    ASSERT_FLOAT_EQ(result, tempclass.expectedSdsdot)
        << "Expected: " << tempclass.expectedSdsdot
        << " Got: " << result;
}

// Tests for dsdot
TEST(DOT, DSDOT1) {
    sycl::queue q = sycl::queue();
    uint64_t N = 10;
    testMixedDOT tempclass(N, 0.0f, q, 42);
    
    float result = syBlast::dsdot(tempclass.N, tempclass.x_double, 1, tempclass.y, 1, q);
    q.wait();
    
    // Using a small epsilon for floating point comparison
    ASSERT_NEAR(result, tempclass.expectedDsdot, 1e-5)
        << "Expected: " << tempclass.expectedDsdot
        << " Got: " << result;
}