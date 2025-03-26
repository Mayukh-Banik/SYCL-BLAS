#include <SYCL-BLAS/blas.hpp>
#include <gtest/gtest.h>
#include <random>
#include <ctime>
#include <iostream>

#define MAX_ELEMENTS 99999999
#define MAX_SCALAR 10

const sycl::queue q = sycl::queue();

template <typename T>
class AxpyFunctionTester
{
public:
	T *x;
	T *y;
	T *expectedY;
	uint64_t N;
	T factor;

	AxpyFunctionTester()
	{
		std::srand(std::time(nullptr));
		this->N = MAX_ELEMENTS;
		this->factor = static_cast<T>(MAX_SCALAR);

		this->x = static_cast<T *>(sycl::malloc_shared(N * sizeof(T), q));
		this->y = static_cast<T *>(sycl::malloc_shared(N * sizeof(T), q));
		this->expectedY = new T[N];

		for (size_t i = 0; i < N; ++i)
		{
			this->x[i] = static_cast<T>(std::rand() % 100);
			this->y[i] = static_cast<T>(std::rand() % 100);
			this->expectedY[i] = this->y[i] + this->x[i] * this->factor;
		}
	}

	~AxpyFunctionTester()
	{
		sycl::free(x, q);
		sycl::free(y, q);
		delete[] expectedY;
	}
};

TEST(Daxpy, Daxpy)
{
	AxpyFunctionTester<double> temp;
	sblas::daxpy(temp.N, temp.factor, temp.x, 1, temp.y, 1, q);

	for (size_t i = 0; i < temp.N; ++i)
	{
		ASSERT_EQ(temp.y[i], temp.expectedY[i]) << "Mismatch at index " << i << " | x[i]: " << temp.x[i]
																		  << " | y[i]: " << temp.y[i]
																		  << " | expectedY[i]: " << temp.expectedY[i]
																		  << " | factor: " << temp.factor;
	}
}

TEST(Saxpy, Saxpy)
{
	AxpyFunctionTester<float> temp;
	sblas::saxpy(temp.N, temp.factor, temp.x, 1, temp.y, 1, q);

	for (size_t i = 0; i < temp.N; ++i)
	{
		ASSERT_EQ(temp.y[i], temp.expectedY[i]) << "Mismatch at index " << i << " | x[i]: " << temp.x[i]
																		  << " | y[i]: " << temp.y[i]
																		  << " | expectedY[i]: " << temp.expectedY[i]
																		  << " | factor: " << temp.factor;
	}
}

TEST(Caxpy, Caxpy)
{
	AxpyFunctionTester<std::complex<float>> temp;
	sblas::caxpy(temp.N, temp.factor, temp.x, 1, temp.y, 1, q);

	for (size_t i = 0; i < temp.N; ++i)
	{
		ASSERT_EQ(temp.y[i], temp.expectedY[i]) << "Mismatch at index " << i << " | x[i]: " << temp.x[i]
																		  << " | y[i]: " << temp.y[i]
																		  << " | expectedY[i]: " << temp.expectedY[i]
																		  << " | factor: " << temp.factor;
	}
}

TEST(Zaxpy, Zaxpy)
{
	AxpyFunctionTester<std::complex<double>> temp;
	sblas::zaxpy(temp.N, temp.factor, temp.x, 1, temp.y, 1, q);

	for (size_t i = 0; i < temp.N; ++i)
	{
		ASSERT_EQ(temp.y[i], temp.expectedY[i]) << "Mismatch at index " << i << " | x[i]: " << temp.x[i]
																		  << " | y[i]: " << temp.y[i]
																		  << " | expectedY[i]: " << temp.expectedY[i]
																		  << " | factor: " << temp.factor;
	}
}