cmake_minimum_required(VERSION 3.28)
project(SYCL-BLAS LANGUAGES CXX)
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create the library target without specifying sources yet
add_library(SYCL-BLAS SHARED)
set_target_properties(SYCL-BLAS PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Add subdirectories
add_subdirectory("Level 1")

target_include_directories(SYCL-BLAS
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)

add_library(SYCL-BLAS::SYCL-BLAS ALIAS SYCL-BLAS)
