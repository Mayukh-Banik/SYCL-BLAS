cmake_minimum_required(VERSION 3.28)


project(syBlast LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(syBlast SHARED)
set(CMAKE_STATIC_LIBRARY_PREFIX "")
set_target_properties(syBlast PROPERTIES 
POSITION_INDEPENDENT_CODE ON
)

# add_subdirectory("Level1")
# add_subdirectory("Database")

target_include_directories(syBlast
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(syBlast PRIVATE
        /O3
        /GL
    )
else()
    set(CMAKE_CXX_FLAGS -O3)
endif()
target_sources(syBlast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/blas.cpp)
add_library(syBlast::syBlast ALIAS syBlast)
