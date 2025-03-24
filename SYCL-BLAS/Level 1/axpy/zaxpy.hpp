#pragma once

#include <sycl/sycl.hpp>
#include <complex>
#include "../../Setup/Setup.hpp"

namespace zaxpyFunction
{
    void defaultZaxpyFunction(uint64_t N, double alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q);

    typedef void (*ZaxpyFunctionPointer)(uint64_t, double, std::complex<double> *, int, std::complex<double> *, int, sycl::queue);

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

        void operator()(uint64_t N, double alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q)
        {
            funcTable[defaultFunction](N, alpha, x, incX, y, incY, q);
        }

        void operator()(uint64_t N, double alpha, std::complex<double> *x, int incX, std::complex<double> *y, int incY, sycl::queue q, int index)
        {
            funcTable[index](N, alpha, x, incX, y, incY, q);
        }
    };

    ZAXPY zaxpyClass;
}