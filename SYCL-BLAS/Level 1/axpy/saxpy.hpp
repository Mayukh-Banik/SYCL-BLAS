#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include "AXPY.hpp"

namespace saxpyFunction
{
#define PARAMETER_LIST uint64_t, float, float *, int, float *, int, sycl::queue
    void defaultSaxpyFunction(PARAMETER_LIST);

    typedef void (*SaxpyFunctionPointer)(PARAMETER_LIST);

    SaxpyFunctionPointer funcTable[] =
        {
            [DEFAULT_FUNCTION_INDEX] = defaultSaxpyFunction,
    };

    int defaultFunction = functionOptimDataBase::functionMap["saxpy"];
#undef PARAMETER_LIST
}