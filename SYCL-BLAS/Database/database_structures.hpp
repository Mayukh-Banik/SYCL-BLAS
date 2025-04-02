#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <sycl/sycl.hpp>
#include <algorithm>

namespace syBlast
{

    constexpr const char *BLAS_NAMES[] =
        {
            "saxpy",
            "daxpy",
            "caxpy",
            "zaxpy",
    };

    enum class BLAS_ENUM_NAMES
    {
        SAXPY = 0,
        DAXPY = 1,
        CAXPY = 2,
        ZAXPY = 3
    };
    ;
    ;

    namespace parameters
    {
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

        constexpr short NUMBER_OF_BLAS_FUNCTIONS = sizeof(BLAS_NAMES) / sizeof(BLAS_NAMES[0]);

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
            DataBaseTable *tables;

            Database()
            {
                tables = new DataBaseTable[NUMBER_OF_BLAS_FUNCTIONS];
                for (int i = 0; i < NUMBER_OF_BLAS_FUNCTIONS; ++i)
                {
                    tables[i] = {"", {}};
                }
                tables[(int)BLAS_ENUM_NAMES::SAXPY] = saxpyTable();
            }

            ~Database()
            {
                delete[] tables;
            }
        };

        class OptimalFunctionParameters
        {
        public:
            Parameters *optimalFunctionParameters;

            OptimalFunctionParameters()
            {
                optimalFunctionParameters = new Parameters[NUMBER_OF_BLAS_FUNCTIONS];
                for (int i = 0; i < NUMBER_OF_BLAS_FUNCTIONS; ++i)
                {
                    optimalFunctionParameters[i] = Parameters();
                }
            }

            OptimalFunctionParameters(const Database &db, std::string device)
            {
                optimalFunctionParameters = new Parameters[NUMBER_OF_BLAS_FUNCTIONS];
                for (int i = 0; i < NUMBER_OF_BLAS_FUNCTIONS; ++i)
                {
                    optimalFunctionParameters[i] = db.tables[i].getParametersForDevice(device);
                }
            }

            ~OptimalFunctionParameters()
            {
                delete[] optimalFunctionParameters;
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