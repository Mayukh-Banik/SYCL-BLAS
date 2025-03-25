#pragma once

#include <sycl/sycl.hpp>
#include <complex>
#include "../../Setup/Setup.hpp"
#include "AXPY.hpp"

namespace zaxpyFunc
{
#define PARAMETER_LIST uint64_t, std::complex<double>, std::complex<double> *, int, std::complex<double> *, int, sycl::queue
    void defaultZaxpyFunction(PARAMETER_LIST);

    typedef void (*ZaxpyFunctionPointer)(PARAMETER_LIST);

    ZaxpyFunctionPointer funcTable[] =
        {
            defaultZaxpyFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["zaxpy"];
#undef PARAMETER_LIST
}