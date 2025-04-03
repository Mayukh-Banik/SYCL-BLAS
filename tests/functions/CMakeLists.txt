set(SOURCE_FILES
    test_axpy.cpp
)

project(Level1 LANGUAGES CXX)

add_executable(tests ${SOURCE_FILES})

target_link_libraries(tests PRIVATE
    gtest
    gtest_main
    syBlast
)