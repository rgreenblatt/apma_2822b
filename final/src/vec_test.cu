#include "vector_uvm.hpp"
#include <vector>
#include <iostream>
#include "cuda_utils.cuh"

__global__ void add_1(int * vec, size_t size) {
  size_t index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index < size) {
    vec[index]++;
  }

}

int main(int /*argc*/, char * /*argv*/[])
{

  std::vector<int, UMAllocator<int>> ints;
  ints.push_back(1);
  ints.push_back(2);
  ints.push_back(3);
  ints.push_back(4);
  add_1<<<4, 1>>>(ints.data(), ints.size());
  cuda_error_chk(cudaDeviceSynchronize());
  std::cout << ints[3] << std::endl;
  
  return 0;
}
