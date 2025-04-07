# CMakeLists.txt for tests

# Add test executable
add_executable(test_temp test_temp.cpp)

# Link GoogleTest to the test executable
target_link_libraries(test_temp gtest gtest_main 
syBlast
)

# Enable testing
enable_testing()

# Add test
add_test(NAME test_temp COMMAND test_temp)