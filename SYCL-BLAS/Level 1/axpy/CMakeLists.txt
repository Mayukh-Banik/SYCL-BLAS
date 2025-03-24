target_sources(SYCL-BLAS
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/saxpy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/daxpy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/caxpy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/zaxpy.cpp
)
