#pragma once

#include <sycl/sycl.hpp>
#include "../../Setup/Setup.hpp"
#include "SCAL.hpp"
#include <complex>
#define NUMBER_OF_FUNCTIONS 1

namespace dscalFunc
{
#define PARAMETER_LIST uint64_t, double, double*, int, sycl::queue
    void defaultDscalFunction(PARAMETER_LIST);
    typedef void (*DscalFunctionPointer)(PARAMETER_LIST);
    DscalFunctionPointer funcTable[] =
        {
            defaultDscalFunction,
    };
    int defaultFunction = functionOptimDataBase::functionMap["dscal"];
#undef PARAMETER_LIST
}