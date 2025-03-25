#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include "SCAL.hpp"
#include <complex>

namespace cscalFunc
{
#define PARAMETER_LIST uint64_t, std::complex<float>, std::complex<float>*, int, sycl::queue
    void defaultCscalFunction(PARAMETER_LIST);
    typedef void (*CscalFunctionPointer)(PARAMETER_LIST);
    CscalFunctionPointer funcTable[] =
        {
            defaultCscalFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["cscal"];
#undef PARAMETER_LIST
}