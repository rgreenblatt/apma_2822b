# Project's name
PROJECT(hw_3 C CXX)

# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wconversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMINIFE_SCALAR=double") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMINIFE_INFO=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_MPI")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMINIFE_FUSED")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMINIFE_DEBUG")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

list(APPEND CUDA_NVCC_FLAGS 
  "-gencode arch=compute_70,code=sm_70;-std=c++11;-O3")
# list( APPEND CUDA_NVCC_FLAGS 
#   "-gencode arch=compute_70,code=sm_70;-std=c++11;-g;-G")

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# The following folder will be included
include_directories(${MPI_INCLUDE_PATH} "${PROJECT_SOURCE_DIR}/include")

cuda_add_executable(
  vec_test 
  "${PROJECT_SOURCE_DIR}/src/vector_uvm.cu"
  "${PROJECT_SOURCE_DIR}/src/vec_test.cu"
  )

if(MPI_COMPILE_FLAGS)
  set_target_properties(vec_test PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(vec_test PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
  
target_link_libraries(vec_test ${MPI_LIBRARIES})

cuda_add_executable(
  miniFE 
  "${PROJECT_SOURCE_DIR}/src/main.cpp"
  "${PROJECT_SOURCE_DIR}/src/vector_uvm.cu"
  "${PROJECT_SOURCE_DIR}/src/cuda_vector_functions.cu"
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
  
target_link_libraries(miniFE ${MPI_LIBRARIES} -lcublas ${CUDA_cusparse_LIBRARY} )
