set(SOURCE_FILES
Complex.cpp
)

project(Setup LANGUAGES CXX)

add_executable(Setup ${SOURCE_FILES})

target_link_libraries(Setup PRIVATE
    gtest
    gtest_main
    SYCL-BLAS
)
