# Project's name
PROJECT(hw_3 C CXX)

# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11 -Wall -Wextra -Wconversion)
add_definitions("-DMINIFE_SCALAR=double" "-DMINIFE_LOCAL_ORDINAL=int"
  "-DMINIFE_GLOBAL_ORDINAL=long long int" -DMINIFE_CSR_MATRIX
  -DMINIFE_INFO=1 -DMINIFE_KERNELS=0 -DHAVE_MPI)
add_definitions(-DMINIFE_FUSED)
# add_definitions(-DMINIFE_DEBUG)
# add_definitions(-g)
add_definitions(-O3 -march=native)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# The following folder will be included
include_directories(${MPI_INCLUDE_PATH} "${PROJECT_SOURCE_DIR}/include")

add_executable(
  miniFE 
  "${PROJECT_SOURCE_DIR}/src/main.cpp"
  "${PROJECT_SOURCE_DIR}/src/box_partition.cpp"
  "${PROJECT_SOURCE_DIR}/src/main.cpp"
  "${PROJECT_SOURCE_DIR}/src/my_timer.cpp"
  "${PROJECT_SOURCE_DIR}/src/param_utils.cpp"
  "${PROJECT_SOURCE_DIR}/src/utils.cpp"
  "${PROJECT_SOURCE_DIR}/src/yaml_doc.cpp"
  "${PROJECT_SOURCE_DIR}/src/yaml_element.cpp"
  )

if(MPI_COMPILE_FLAGS)
  set_target_properties(miniFE PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(miniFE PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
  
target_link_libraries(miniFE ${MPI_LIBRARIES})
