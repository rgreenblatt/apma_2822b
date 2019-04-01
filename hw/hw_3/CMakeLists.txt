# Project's name
PROJECT(hw_3 C CXX)

# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11 -Wall -Wextra -Wconversion -DEIGEN)
# add_definitions(-g)
add_definitions(-O3 -march=native)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# The following folder will be included
include_directories(${MKL_INCLUDE_DIRS} ${MPI_INCLUDE_PATH} "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/eigen/")

add_executable(app 
    "${PROJECT_SOURCE_DIR}/src/main.cpp"
)

if(MPI_COMPILE_FLAGS)
  set_target_properties(app PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(app PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
  
target_link_libraries(app ${MPI_LIBRARIES} ${MKL_LIBRARIES})
