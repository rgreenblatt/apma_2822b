#include "cuda_utils.hpp"

namespace miniFE {
void select_cuda_device(int mpi_rank) {
  int device_count;
  cudaGetDeviceCount(&device_count);
  cudaSetDevice(mpi_rank % device_count);
}
} // namespace miniFE
