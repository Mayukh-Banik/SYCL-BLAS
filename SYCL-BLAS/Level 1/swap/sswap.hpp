#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#define NUMBER_OF_FUNCTIONS 1

namespace swapFunc
{
#define PARAMETER_LIST uint64_t, float*, int, float*, int, sycl::queue
    void defaultSswapFunction(PARAMETER_LIST);
    typedef void (*SswapFunctionPointer)(PARAMETER_LIST);
    SswapFunctionPointer funcTable[] =
        {
            defaultSswapFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["swap"];
#undef PARAMETER_LIST
}