#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include "SCAL.hpp"
#include <complex>
#define NUMBER_OF_FUNCTIONS 1

namespace csscalFunc
{
#define PARAMETER_LIST uint64_t, float, std::complex<float>*, int, sycl::queue
    void defaultCsscalFunction(PARAMETER_LIST);
    typedef void (*CsscalFunctionPointer)(PARAMETER_LIST);
    CsscalFunctionPointer funcTable[] =
        {
            defaultCsscalFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["csscal"];
#undef PARAMETER_LIST
}