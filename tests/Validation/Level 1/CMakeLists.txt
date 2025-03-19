set(SOURCE_FILES
    axpy.cpp
    copy.cpp
    scal.cpp
    swap.cpp
    nrm2.cpp
)

project(Level1 LANGUAGES CXX)

add_executable(tests ${SOURCE_FILES})

target_link_libraries(tests PRIVATE
    gtest
    gtest_main
    SYCL-BLAS
)
