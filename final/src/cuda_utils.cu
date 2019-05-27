#include "cuda_utils.hpp"

namespace miniFE {
void select_cuda_device(int mpi_rank) {
  int device_count;
  cudaGetDeviceCount(&device_count);
  cudaSetDevice(mpi_rank % device_count);
}

__global__ void copy_to_buffer_kernel(MINIFE_SCALAR *buffer, const MINIFE_SCALAR *from,
                    const int *elements_to_copy,size_t total_to_be_sent) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < total_to_be_sent) {
    buffer[i] = from[elements_to_copy[i]];
  }
}

void copy_to_buffer(MINIFE_SCALAR *buffer, const MINIFE_SCALAR *from,
                    const int *elements_to_copy, size_t total_to_be_sent) {
  const size_t thread_num = 256;
  copy_to_buffer_kernel<<<(total_to_be_sent + thread_num - 1) / thread_num,
                          thread_num>>>(buffer, from, elements_to_copy,
                                        total_to_be_sent);
}
} // namespace miniFE
