# Project's name
PROJECT(lecture_1 C CXX)

cmake_minimum_required(VERSION 3.12)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(CUDA_NVCC_FLAGS "-std=c++11 -O3")
# set(CUDA_NVCC_FLAGS "-std=c++11 -g")

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


cuda_add_executable(app 
    "${PROJECT_SOURCE_DIR}/src/spmv.cu"
)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
