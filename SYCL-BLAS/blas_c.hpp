#pragma once

#include "blas.hpp"
#include <complex>

// Define a macro for exporting symbols
#if defined(_WIN32) || defined(_WIN64)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif

extern "C"
{

    EXPORT_SYMBOL void saxpy(int n, float a, const float *x, int incx, float *y, int incy)
    {
        syBlast::saxpy(n, a, x, incx, y, incy);
    }
    
    EXPORT_SYMBOL void daxpy(int n, double a, const double *x, int incx, double *y, int incy);
    EXPORT_SYMBOL void caxpy(int n, std::complex<float> a, const std::complex<float> *x, int incx, std::complex<float> *y, int incy);
    EXPORT_SYMBOL void zaxpy(int n, std::complex<double> a, const std::complex<double> *x, int incx, std::complex<double> *y, int incy);
}