if(UNIX)
    set(CMAKE_CXX_COMPILER icx)
else()
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

cmake_minimum_required (VERSION 3.28)

project(VectorAdd CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(extern/SyComplex)
add_subdirectory (syBlast)

option(ENABLE_TESTING "Enable testing" OFF)

if(ENABLE_TESTING)
    set(GOOGLETEST_DIR ${CMAKE_SOURCE_DIR}/extern/googletest)
    add_subdirectory(${GOOGLETEST_DIR} ${CMAKE_BINARY_DIR}/googletest)
    add_subdirectory(tests)
    enable_testing()
endif()