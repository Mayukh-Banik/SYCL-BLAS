#pragma once

#include <sycl/sycl.hpp>
#include <complex>
#include "../../Setup/Setup.hpp"

namespace saxpyFunction
{
    void defaultSaxpyFunction(uint64_t N, float alpha, float *x, int incX, float *y, int incY, sycl::queue q);

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