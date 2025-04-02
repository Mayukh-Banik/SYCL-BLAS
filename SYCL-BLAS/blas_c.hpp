#pragma once

#include "blas.hpp"
#include <complex>

// Define a macro for exporting symbols
#if defined(_WIN32) || defined(_WIN64)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL 
#endif

extern "C"
{

    EXPORT_SYMBOL void saxpy(int n, float a, const float *x, int incx, float *y, int incy);
    
}