#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <sycl/sycl.hpp>
#include <algorithm>

namespace syBlast
{
    /**
     * @brief Array of BLAS function names as constant strings.
     *
     * This array contains the names of the BLAS (Basic Linear Algebra Subprograms)
     * functions supported by the database. Each name corresponds to a specific
     * BLAS function, and the order of the names matches the order of the
     * BLAS_ENUM_NAMES enumeration.
     *
     */
    constexpr const char *BLAS_NAMES[] =
        {
            "saxpy",
            "daxpy",
            "caxpy",
            "zaxpy",
            "sscal",
            "dscal",
            "cscal",
            "zscal",
            "csscal",
            "zsscal"
    };

    /**
     * @brief Enumeration of BLAS function names.
     *
     * This enumeration defines the identifiers for the BLAS
     * functions supported by the database. Each enumerator corresponds to a specific BLAS function,
     * and its value matches the index of the function name in the `BLAS_NAMES` array.
     */
    enum class BLAS_ENUM_NAMES
    {
        SAXPY = 0,
        DAXPY = 1,
        CAXPY = 2,
        ZAXPY = 3,
        SSCAL = 4,
        DSCAL = 5,
        CSCAL = 6,
        ZSCAL = 7,
        CSSCAL = 8,
        ZSSCAL = 9,
    };;

    namespace parameters
    {
        /**
         * @brief Function Parameters for all syBlast Functions
         * 
         * FunctionParameters stores all the parameters that can be used by a BLAS function.
         * Parameters[0] refers to function index. Each BLAS function can use multiple different functions (recursion vs iterative methods, etc) 
         * and thus P[0] is always the index, with P[0] == 0 always being the base case with no optimizations, that will work on any platform.
         * The parameters used by each funtion (AXPY, SCAL, etc) will be noted at the top of their respective .cpp files or "syBlast/Level*\/*"
         */
        class FunctionParameters
        {
        public:
            uint64_t Parameters[16];

            constexpr FunctionParameters(
                uint64_t p0 = 0, uint64_t p1 = 0, uint64_t p2 = 0, uint64_t p3 = 0,
                uint64_t p4 = 0, uint64_t p5 = 0, uint64_t p6 = 0, uint64_t p7 = 0,
                uint64_t p8 = 0, uint64_t p9 = 0, uint64_t p10 = 0, uint64_t p11 = 0,
                uint64_t p12 = 0, uint64_t p13 = 0, uint64_t p14 = 0, uint64_t p15 = 0) : Parameters{p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15}
            {
            }

            constexpr uint64_t &operator[](size_t index) noexcept
            {
                return Parameters[index];
            }

            constexpr const uint64_t &operator[](size_t index) const noexcept
            {
                return Parameters[index];
            }
        };
    }

    namespace database
    {

        /**
         * This will eventually turn into a #define, but while all functions aren't implemented, this will stay.
         */
        constexpr unsigned long NUMBER_OF_BLAS_FUNCTIONS = sizeof(BLAS_NAMES) / sizeof(BLAS_NAMES[0]);

        using DeviceName = std::string;
        using Parameters = parameters::FunctionParameters;

        typedef struct DataBaseEntry
        {
            std::vector<DeviceName> names;
            Parameters parameters;

            bool containsDevice(const DeviceName &device) const
            {
                return std::find(names.begin(), names.end(), device) != names.end();
            }

        } DataBaseEntry;

        typedef struct DataBaseTable
        {
            std::string FunctionName;
            std::vector<DataBaseEntry> entries;

            Parameters getParametersForDevice(const DeviceName &device) const
            {
                for (const auto &entry : entries)
                {
                    if (entry.containsDevice(device))
                    {
                        return entry.parameters;
                    }
                }
                return Parameters();
            }
        } DataBaseTable;

        DataBaseTable saxpyTable();

        class Database
        {
        public:
            DataBaseTable tables[NUMBER_OF_BLAS_FUNCTIONS];

            Database()
            {
                for (int i = 0; i < NUMBER_OF_BLAS_FUNCTIONS; ++i)
                {
                    tables[i] = {"", {}};
                }
                tables[(int)BLAS_ENUM_NAMES::SAXPY] = saxpyTable();
            }
        };

        class OptimalFunctionParameters
        {
        public:
            Parameters optimalFunctionParameters[NUMBER_OF_BLAS_FUNCTIONS];

            OptimalFunctionParameters()
            {
                for (int i = 0; i < NUMBER_OF_BLAS_FUNCTIONS; ++i)
                {
                    optimalFunctionParameters[i] = Parameters();
                }
            }

            OptimalFunctionParameters(const Database &db, std::string device)
            {
                for (int i = 0; i < NUMBER_OF_BLAS_FUNCTIONS; ++i)
                {
                    optimalFunctionParameters[i] = db.tables[i].getParametersForDevice(device);
                }
            }

            Parameters &operator[](BLAS_ENUM_NAMES func) noexcept
            {
                return optimalFunctionParameters[static_cast<short>(func)];
            }

            const Parameters &operator[](BLAS_ENUM_NAMES func) const noexcept
            {
                return optimalFunctionParameters[static_cast<short>(func)];
            }
        };

    }
}