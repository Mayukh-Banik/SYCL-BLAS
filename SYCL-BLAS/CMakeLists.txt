cmake_minimum_required(VERSION 3.28)
project(SYCL-BLAS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



add_library(SYCL-BLAS SHARED)
set_target_properties(SYCL-BLAS PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory("Level 1")

target_include_directories(SYCL-BLAS
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(SYCL-BLAS PRIVATE
        /O2
        /GL
    )
else()
set(CMAKE_CXX_FLAGS -O3)
    # target_compile_options(SYCL-BLAS PRIVATE
    #     -O3
    # )
endif()
add_library(SYCL-BLAS::SYCL-BLAS ALIAS SYCL-BLAS)
