#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include "SCAL.hpp"
#include <complex>

namespace zscalFunc
{
#define PARAMETER_LIST uint64_t, std::complex<double>, std::complex<double>*, int, sycl::queue
    void defaultZscalFunction(PARAMETER_LIST);
    typedef void (*ZscalFunctionPointer)(PARAMETER_LIST);
    ZscalFunctionPointer funcTable[] =
        {
            defaultZscalFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["zscal"];
#undef PARAMETER_LIST
}