file(GLOB_RECURSE SOURCE_FILES
    "*.cpp"
)

project(tests LANGUAGES CXX)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE
    gtest
    gtest_main
    SYCL-BLAS
)