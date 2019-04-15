# Project's name
PROJECT(hw_3 C CXX)

# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11 -Wall -Wextra -Wconversion 
  -fopenmp-targets=nvptx64-nvidia-cuda)
# add_definitions(-g)
add_definitions(-O3 -march=native)

find_package(OpenMP REQUIRED)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}/include")

add_executable(app 
    "${PROJECT_SOURCE_DIR}/src/main.cpp"
)
