#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include <complex>
#define NUMBER_OF_FUNCTIONS 1

namespace ccopyFunc
{
#define PARAMETER_LIST uint64_t, std::complex<float>*, int, std::complex<float>*, int, sycl::queue
    void defaultCcopyFunction(PARAMETER_LIST);
    typedef void (*CcopyFunctionPointer)(PARAMETER_LIST);
    CcopyFunctionPointer funcTable[] =
        {
            defaultCcopyFunction,
    };

    int defaultFunction = functionOptimDataBase::functionMap["ccopy"];
#undef PARAMETER_LIST
}