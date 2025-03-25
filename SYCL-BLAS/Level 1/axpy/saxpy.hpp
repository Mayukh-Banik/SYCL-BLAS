#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include "AXPY.hpp"
#define NUMBER_OF_FUNCTIONS 1

namespace saxpyFunc
{
#define PARAMETER_LIST uint64_t, float, float *, int, float *, int, sycl::queue
    void defaultSaxpyFunction(PARAMETER_LIST);

    typedef void (*SaxpyFunctionPointer)(PARAMETER_LIST);

    SaxpyFunctionPointer funcTable[] =
        {
            defaultSaxpyFunction,
    };

    int defaultFunction = functionOptimDataBase::functionMap["saxpy"];
#undef PARAMETER_LIST
}