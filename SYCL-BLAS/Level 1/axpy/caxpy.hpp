#pragma once

#include <sycl/sycl.hpp>
#include <complex>
#include "../../Setup/Setup.hpp"

namespace caxpyFunction
{
    void defaultCaxpyFunction(uint64_t N, float alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q);

    typedef void (*CaxpyFunctionPointer)(uint64_t, float, std::complex<float> *, int, std::complex<float> *, int, sycl::queue);

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

        void operator()(uint64_t N, float alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q)
        {
            funcTable[defaultFunction](N, alpha, x, incX, y, incY, q);
        }

        void operator()(uint64_t N, float alpha, std::complex<float> *x, int incX, std::complex<float> *y, int incY, sycl::queue q, int index)
        {
            funcTable[index](N, alpha, x, incX, y, incY, q);
        }
    };

    CAXPY caxpyClass;
}