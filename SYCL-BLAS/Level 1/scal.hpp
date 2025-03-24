#pragma once

#include <sycl/sycl.hpp>
#include <complex>
#include "../Setup/Setup.hpp"

namespace scalFunctions
{
    namespace saxpyFunction
    {
        void defaultSaxpyFunction(uint64_t N, float alpha, float *x, int incX, float *y, int incY, sycl::queue q)
        {
            q.submit(
                [&](sycl::handler &handler)
                {
                    handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                         {
                   int64_t Xindex = i[0] * incX;
                   int64_t Yindex = i[0] * incY;
                   y[Yindex] += alpha * x[Xindex]; });
                });
        }

        typedef void (*SaxpyFunctionPointer)(uint64_t, float, float *, int, float *, int, sycl::queue);

        SaxpyFunctionPointer funcTable[1] =
            {
                defaultSaxpyFunction,
        };
        class SAXPY
        {
        private:
            int defaultFunction = 0;

        public:
            SAXPY()
            {
                this->defaultFunction = functionOptimDataBase::functionMap["saxpy"];
            }

            void operator()(uint64_t N, float alpha, float *x, int incX, float *y, int incY, sycl::queue q)
            {
                funcTable[defaultFunction](N, alpha, x, incX, y, incY, q);
            }

            void operator()(uint64_t N, float alpha, float *x, int incX, float *y, int incY, sycl::queue q, int index)
            {
                funcTable[index](N, alpha, x, incX, y, incY, q);
            }
        };

        SAXPY saxpyClass;
    }

    namespace daxpyFunction
    {
        void defaultDaxpyFunction(uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q)
        {
            q.submit(
                [&](sycl::handler &handler)
                {
                    handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                         {
                   int64_t Xindex = i[0] * incX;
                   int64_t Yindex = i[0] * incY;
                   y[Yindex] += alpha * x[Xindex]; });
                });
        }

        typedef void (*DaxpyFunctionPointer)(uint64_t, double, double *, int, double *, int, sycl::queue);

        DaxpyFunctionPointer funcTable[1] =
            {
                defaultSaxpyFunction,
        };
        class DAXPY
        {
        private:
            int defaultFunction = 0;

        public:
            DAXPY()
            {
                this->defaultFunction = functionOptimDataBase::functionMap["daxpy"];
            }

            void operator()(uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q)
            {
                funcTable[defaultFunction](N, alpha, x, incX, y, incY, q);
            }

            void operator()(uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q, int index)
            {
                funcTable[index](N, alpha, x, incX, y, incY, q);
            }
        };

        DAXPY daxpyClass;
    }

    namespace caxpyFunction
    {
        void defaultCaxpyFunction(uint64_t N, std::complex<float> alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q)
        {
            q.submit(
                [&](sycl::handler &handler)
                {
                    handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                         {
                       int64_t Xindex = i[0] * incX;
                       int64_t Yindex = i[0] * incY;
                       y[Yindex] += alpha * x[Xindex]; });
                });
        }

        typedef void (*CaxpyFunctionPointer)(uint64_t, std::complex<float>, std::complex<float> *, int, std::complex<float> *, int, sycl::queue);

        CaxpyFunctionPointer funcTable[1] =
            {
                defaultCaxpyFunction,
        };

        class CAXPY
        {
        private:
            int defaultFunction = 0;

        public:
            CAXPY()
            {
                this->defaultFunction = functionOptimDataBase::functionMap["caxpy"];
            }

            void operator()(uint64_t N, std::complex<float> alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q)
            {
                funcTable[defaultFunction](N, alpha, x, incX, y, incY, q);
            }

            void operator()(uint64_t N, std::complex<float> alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q, int index)
            {
                funcTable[index](N, alpha, x, incX, y, incY, q);
            }
        };

        CAXPY caxpyClass;
    }

    namespace zaxpyFunction
    {
        void defaultZaxpyFunction(uint64_t N, std::complex<double> alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q)
        {
            q.submit(
                [&](sycl::handler &handler)
                {
                    handler.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                         {
                       int64_t Xindex = i[0] * incX;
                       int64_t Yindex = i[0] * incY;
                       y[Yindex] += alpha * x[Xindex]; });
                });
        }

        typedef void (*ZaxpyFunctionPointer)(uint64_t, std::complex<double>, std::complex<double> *, int, std::complex<double> *, int, sycl::queue);

        ZaxpyFunctionPointer funcTable[1] =
            {
                defaultZaxpyFunction,
        };

        class ZAXPY
        {
        private:
            int defaultFunction = 0;

        public:
            ZAXPY()
            {
                this->defaultFunction = functionOptimDataBase::functionMap["zaxpy"];
            }

            void operator()(uint64_t N, std::complex<double> alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q)
            {
                funcTable[defaultFunction](N, alpha, x, incX, y, incY, q);
            }

            void operator()(uint64_t N, std::complex<double> alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q, int index)
            {
                funcTable[index](N, alpha, x, incX, y, incY, q);
            }
        };

        ZAXPY zaxpyClass;
    }
}
