#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include <complex>
#define NUMBER_OF_FUNCTIONS 1

namespace dswapFunc
{
#define PARAMETER_LIST uint64_t, double*, int, double*, int, sycl::queue
    void defaultDswapFunction(PARAMETER_LIST);
    typedef void (*DswapFunctionPointer)(PARAMETER_LIST);
    DswapFunctionPointer funcTable[] =
        {
            defaultDswapFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["dswap"];
#undef PARAMETER_LIST
}