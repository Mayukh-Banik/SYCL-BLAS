cmake_minimum_required(VERSION 3.28)

project(syBlast)

add_subdirectory(syBlast)

option(ENABLE_TESTING "Enable testing" OFF)

if(ENABLE_TESTING)
    add_subdirectory(extern/googletest)
    add_subdirectory(tests)
endif()

