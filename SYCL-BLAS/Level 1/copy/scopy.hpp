#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include <complex>
#define NUMBER_OF_FUNCTIONS 1

namespace scopyFunc
{
#define PARAMETER_LIST uint64_t, float*, int, float*, int, sycl::queue
    void defaultScopyFunction(PARAMETER_LIST);
    typedef void (*ScopyFunctionPointer)(PARAMETER_LIST);
    ScopyFunctionPointer funcTable[] =
        {
            defaultScopyFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["scopy"];
#undef PARAMETER_LIST
}