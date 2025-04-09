set(EXECUTABLE_NAME test_Validation)

add_executable(${EXECUTABLE_NAME} 
  test_vectorScaling.cpp
  test_dot.cpp
)

target_link_libraries(${EXECUTABLE_NAME}
  gtest
  gtest_main
  syBlast
)

enable_testing()

add_test(NAME ${EXECUTABLE_NAME} COMMAND ${EXECUTABLE_NAME})