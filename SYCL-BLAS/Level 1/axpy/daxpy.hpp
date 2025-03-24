#pragma once

#include <sycl/sycl.hpp>
#include <complex>
#include "../../Setup/Setup.hpp"

namespace daxpyFunction
{
    void defaultDaxpyFunction(uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q);

    typedef void (*DaxpyFunctionPointer)(uint64_t, double, double *, int, double *, int, sycl::queue);

    DaxpyFunctionPointer funcTable[1] =
        {
            defaultDaxpyFunction,
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