target_sources(SYCL-BLAS
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ccopy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dcopy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scopy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/zcopy.cpp
    )