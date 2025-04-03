#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <complex>
#include "Database/database.hpp"

#if defined(_WIN32) || defined(_WIN64)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL 
#endif

/**
 * @mainpage SYCL-BLAS Documentation
 *
 * @section intro_sec Introduction
 *
 * An open source implementation of BLAS in SYCL with Tuners for multiple platforms.
 *
 * @section install_sec Installation
 *
 * 1. SYCL Compiler
 * 2. CMake
 * 3. SYCL capable device (everything except AMD CPUS and new AMD GPUs).
 *
 * @section usage_sec Usage
 *
 * @ref sblas
 */

/**
 * @namespace syBlast
 *
 * Location for all NetLib conformant C style functions from library SYCL-BLAS.
 * All functions located here will follow NetLib style naming and function parameters.
 */
extern "C"
{
    namespace syBlast
    {
        /**
         * This is documentation
         */
        void saxpy(const uint64_t N, const float alpha, const float *x, const int incx, float *y, const int incy, sycl::queue q = sycl::queue(), database::Parameters p = database::FuncParamDB[BLAS_ENUM_NAMES::SAXPY]);

    }
}