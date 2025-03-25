#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#define NUMBER_OF_FUNCTIONS 1

namespace dcopyFunc
{
#define PARAMETER_LIST uint64_t, double*, int, double*, int, sycl::queue
    void defaultDcopyFunction(PARAMETER_LIST);
    typedef void (*DcopyFunctionPointer)(PARAMETER_LIST);
    DcopyFunctionPointer funcTable[] =
        {
            defaultDcopyFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["dcopy"];
#undef PARAMETER_LIST
}