#pragma once

#include "database_structures.hpp"
#include <sycl/sycl.hpp>

namespace syBlast
{
    namespace database
    {
        static const Database DB;
        static const OptimalFunctionParameters FuncParamDB(DB, sycl::device{}.get_info<sycl::info::device::name>());
    }
}