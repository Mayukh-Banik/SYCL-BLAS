set(SOURCE_FILES 
    kernels/vectorScaling.cpp
    kernels/dot.cpp
    kernels/nrmsetc.cpp
    )

set(TARGET_NAME syBlast)

if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
set(LINK_FLAGS "-fsycl")

add_library(${TARGET_NAME} STATIC ${SOURCE_FILES})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")


target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)

add_library(${TARGET_NAME}::${TARGET_NAME} ALIAS ${TARGET_NAME})