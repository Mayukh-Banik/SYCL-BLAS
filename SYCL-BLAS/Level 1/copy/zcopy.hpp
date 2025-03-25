#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include <complex>

#define NUMBER_OF_FUNCTIONS 1

namespace zcopyFunc
{
#define PARAMETER_LIST uint64_t, std::complex<double>*, int, std::complex<double>*, int, sycl::queue
    void defaultZcopyFunction(PARAMETER_LIST);
    typedef void (*ZcopyFunctionPointer)(PARAMETER_LIST);
    ZcopyFunctionPointer funcTable[] =
        {
            defaultZcopyFunction,
    };

    int defaultFunction = functionOptimDataBase::functionMap["zcopy"];
#undef PARAMETER_LIST
}