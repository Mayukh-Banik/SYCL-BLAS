#pragma once

#include <sycl/sycl.hpp>
#include <complex>
#include "../../Setup/Setup.hpp"
#include "AXPY.hpp"

namespace caxpyFunction
{
    #define PARAMETER_LIST uint64_t, float, std::complex<float> *, int, std::complex<float> *, int, sycl::queue
    void defaultCaxpyFunction(PARAMETER_LIST);

    typedef void (*CaxpyFunctionPointer)(PARAMETER_LIST);

    CaxpyFunctionPointer funcTable[] =
        {
            defaultCaxpyFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["caxpy"];

    #undef PARAMETER_LIST
}