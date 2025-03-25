#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include <complex>

#define NUMBER_OF_FUNCTIONS 1

namespace cswapFunc
{
#define PARAMETER_LIST uint64_t, std::complex<float>*, int, std::complex<float>*, int, sycl::queue
    void defaultCswapFunction(PARAMETER_LIST);
    typedef void (*CswapFunctionPointer)(PARAMETER_LIST);
    CswapFunctionPointer funcTable[] =
        {
            defaultCswapFunction,
    };

    int defaultFunction = functionOptimDataBase::functionMap["cswap"];
#undef PARAMETER_LIST
}