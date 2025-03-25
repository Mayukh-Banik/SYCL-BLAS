#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include "SCAL.hpp"
#include <complex>
#define NUMBER_OF_FUNCTIONS 1

namespace zdscalFunc
{
#define PARAMETER_LIST uint64_t, double, std::complex<double>*, int, sycl::queue
    void defaultZdscalFunction(PARAMETER_LIST);
    typedef void (*ZdscalFunctionPointer)(PARAMETER_LIST);
    ZdscalFunctionPointer funcTable[] =
        {
            defaultZdscalFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["zdscal"];
#undef PARAMETER_LIST
}