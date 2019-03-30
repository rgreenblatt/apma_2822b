# Project's name
PROJECT(lecture_1 C CXX)

# Specify the minimum version for CMake

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "-std=c++11 -O3")

cuda_add_executable(unmanaged 
    "${PROJECT_SOURCE_DIR}/src/unmanaged.cu"
)

cuda_add_executable(managed 
    "${PROJECT_SOURCE_DIR}/src/managed.cu"
)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
