# Project's name
PROJECT(lecture_1 C CXX)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

list( APPEND CUDA_NVCC_FLAGS "-arch=sm_50;-std=c++11;-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")
# set(CUDA_NVCC_FLAGS "-std=c++11 -g")

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/include")

cuda_add_executable(app ${CMAKE_CURRENT_SOURCE_DIR}/src/spmv.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/methods.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cu)


target_link_libraries( app ${CUDA_cusparse_LIBRARY} )
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
