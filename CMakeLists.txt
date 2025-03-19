cmake_minimum_required(VERSION 3.28)

project(SYCL-BLAS)

add_subdirectory(SYCL-BLAS)

option(ENABLE_TESTING "Enable testing" OFF)

if(ENABLE_TESTING)
    add_subdirectory(extern/googletest)
    add_subdirectory(tests)
endif()