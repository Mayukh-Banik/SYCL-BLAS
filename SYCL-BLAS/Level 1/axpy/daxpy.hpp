#pragma once

#include <sycl/sycl.hpp>
#include <complex>
#include "../../Setup/Setup.hpp"
#include "AXPY.hpp"
#define NUMBER_OF_FUNCTIONS 1

namespace daxpyFunc
{
    #define PARAMETER_LIST uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q
    void defaultDaxpyFunction(PARAMETER_LIST);

    typedef void (*DaxpyFunctionPointer)(PARAMETER_LIST);

    DaxpyFunctionPointer funcTable[] =
        {
            defaultDaxpyFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["daxpy"];
    #undef PARAMETER_LIST
}