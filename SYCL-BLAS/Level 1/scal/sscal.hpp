#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include "SCAL.hpp"
#include <complex>
#define NUMBER_OF_FUNCTIONS 1

namespace SscalFunc
{
#define PARAMETER_LIST uint64_t, float, float*, int, sycl::queue
    void defaultSscalFunction(PARAMETER_LIST);
    typedef void (*SscalFunctionPointer)(PARAMETER_LIST);
    SscalFunctionPointer funcTable[] =
        {
            defaultSscalFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["sscal"];
#undef PARAMETER_LIST
}