# Project's name
PROJECT(lecture_1 C CXX)

# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_definitions(-std=c++11 -Wall -Wextra -Wconversion -O3)

add_executable(app 
    "${PROJECT_SOURCE_DIR}/src/main.cpp"
)

target_link_libraries(app ${MPI_LIBRARIES})

include_directories(${MPI_INCLUDE_PATH} "${PROJECT_SOURCE_DIR}/include")

if(MPI_COMPILE_FLAGS)
  set_target_properties(app PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(app PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
