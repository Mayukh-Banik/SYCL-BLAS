#pragma once

#include <unordered_map>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace functionOptimDataBase
{

    const std::vector<std::string> BLAS_NAMES =
        {
            // Level 1 BLAS
            "sasum", "dasum", "casum", "zasum",
            "saxpy", "daxpy", "caxpy", "zaxpy",
            "scopy", "dcopy", "ccopy", "zcopy",
            "sdot", "ddot", "cdotu", "zdotu", "cdotc", "zdotc",
            "snrm2", "dnrm2", "scnrm2", "dznrm2",
            "srot", "drot", "csrot", "zdrot",
            "srotg", "drotg", "crotg", "zrotg",
            "srotm", "drotm",
            "srotmg", "drotmg",
            "sscal", "dscal", "cscal", "zscal", "csscal", "zdscal",
            "sswap", "dswap", "cswap", "zswap",
            "isamax", "idamax", "icamax", "izamax",

            // Level 2 BLAS
            "sgemv", "dgemv", "cgemv", "zgemv",
            "sger", "dger", "cgeru", "zgeru", "cgerc", "zgerc",
            "ssymv", "dsymv", "chemv", "zhemv",
            "ssyr", "dsyr", "cher", "zher",
            "ssyr2", "dsyr2", "cher2", "zher2",
            "strmv", "dtrmv", "ctrmv", "ztrmv",
            "strsv", "dtrsv", "ctrsv", "ztrsv",
            "stbmv", "dtbmv", "ctbmv", "ztbmv",
            "stbsv", "dtbsv", "ctbsv", "ztbsv",
            "stpmv", "dtpmv", "ctpmv", "ztpmv",
            "stpsv", "dtpsv", "ctpsv", "ztpsv",
            "sspmv", "dspmv", "chpmv", "zhpmv",
            "sspr", "dspr", "chpr", "zhpr",
            "sspr2", "dspr2", "chpr2", "zhpr2",

            // Level 3 BLAS
            "sgemm", "dgemm", "cgemm", "zgemm",
            "ssymm", "dsymm", "csymm", "zsymm",
            "ssyrk", "dsyrk", "csyrk", "zsyrk",
            "ssyr2k", "dsyr2k", "csyr2k", "zsyr2k",
            "strmm", "dtrmm", "ctrmm", "ztrmm",
            "strsm", "dtrsm", "ctrsm", "ztrsm",
            "chemm", "zhemm",
            "cherk", "zherk",
            "cher2k", "zher2k"};

    const std::unordered_map<std::string, std::unordered_map<std::string, int>> optimizedFunctionDatabase =
        {
            // {"AdaptiveCpp OpenMP host device",
            //  {{"saxpy", 0}}},
    };
    class OptimizedFunctionHashMap
    {
    private:
        std::unordered_map<std::string, int> optimalFunctionalHashMap;
        std::string defaultDevice;

    public:
        OptimizedFunctionHashMap()
        {
            sycl::queue q;
            defaultDevice = q.get_device().get_info<sycl::info::device::name>();
            for (const std::string &a : BLAS_NAMES)
            {
                optimalFunctionalHashMap[a] = 0;
            }

            try
            {
                const auto &temp = optimizedFunctionDatabase.at(defaultDevice);
                for (const auto &pair : temp)
                {
                    optimalFunctionalHashMap[pair.first] = pair.second;
                }
            }
            catch (const std::exception &e)
            {
            }
        }

        int operator[](std::string functionName)
        {
            return optimalFunctionalHashMap[functionName];
        }
    };

    static OptimizedFunctionHashMap functionMap;
}