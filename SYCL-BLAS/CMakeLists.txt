cmake_minimum_required(VERSION 3.28)

add_library(SYCL-BLAS INTERFACE)

target_include_directories(SYCL-BLAS 
    INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)

add_library(SYCL-BLAS::SYCL-BLAS ALIAS SYCL-BLAS)