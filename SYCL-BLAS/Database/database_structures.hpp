#pragma once

#include <sycl/sycl.hpp>
#include <vector>

namespace syBlast
{
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

            constexpr uint64_t& operator[](size_t index) noexcept
            {
                return Parameters[index];
            }
        
            constexpr const uint64_t& operator[](size_t index) const noexcept
            {
                return Parameters[index];
            }
        };
    }

    namespace database
    {
        using DeviceName = std::string;
        using Parameters = parameters::FunctionParameters;

        typedef struct DataBaseEntry
        {
            std::vector<DeviceName> names;
            Parameters parameters;
        }DataBaseEntry;

        struct DataBaseTable
        {
            std::string FunctionName;
            std::vector<DataBaseEntry> entries;
        };
    }
}