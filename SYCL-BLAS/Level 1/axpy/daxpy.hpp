#pragma once

#include <sycl/sycl.hpp>
#include <complex>
#include "../../Setup/Setup.hpp"
#include "AXPY.hpp"

namespace daxpyFunction
{
    #define PARAMETER_LIST uint64_t N, double alpha, double *x, int incX, double *y, int incY, sycl::queue q
    void defaultDaxpyFunction(PARAMETER_LIST);

    typedef void (*DaxpyFunctionPointer)(PARAMETER_LIST);

    DaxpyFunctionPointer funcTable[1] =
        {
            defaultDaxpyFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["daxpy"];
    #undef PARAMETER_LIST
}