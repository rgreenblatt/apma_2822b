#pragma once
#include "methods.h"
#include <chrono>
#include <stdio.h>

#define cuda_sparse_error_chk(ans)                                             \
  { cuda_sparse_assert((ans), __FILE__, __LINE__); }
inline void cuda_sparse_assert(cusparseStatus_t code, const char *file,
                               int line) {
  if (code != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "Cuda sparse error %d %s %d\n", code, file, line);
    exit(code);
  }
}

#define cuda_error_chk(ans)                                                    \
  { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

namespace chr = std::chrono;
using h_clock = chr::high_resolution_clock;
enum class MemoryType { Host, Device, Unified };

void time_function(int iterations, SpMvMethod &method, double *times,
                   bool is_cuda);

void print_result(double *times, int iterations, const char *name);

template <class T> void allocate_vector(T *&A, int n, MemoryType memory_type) {
  switch (memory_type) {
  case MemoryType::Host:
    A = new T[n];
    break;
  case MemoryType::Device:
    cuda_error_chk(cudaMalloc(&A, n * sizeof(T)));
    break;
  case MemoryType::Unified:
    cuda_error_chk(cudaMallocManaged(&A, n * sizeof(T)));
    break;
  }
}
template <class T> size_t allocate_matrix_device(T *&A, int n, int m) {
  size_t pitch;
  cuda_error_chk(cudaMallocPitch(&A, &pitch, m * sizeof(T), n));
  return pitch;
}

template <class T>
void allocate_matrix(T **&A, int n, int m, MemoryType memory_type) {
  switch (memory_type) {
  case MemoryType::Host:
    A = new T *[m];
    A[0] = new T[n * m];
    break;
  case MemoryType::Device:
    cuda_error_chk(cudaMallocManaged(&A, m * sizeof(T *)));
    cuda_error_chk(cudaMalloc(&(A[0]), n * m * sizeof(T)));
    break;
  case MemoryType::Unified:
    cuda_error_chk(cudaMallocManaged(&A, m * sizeof(T *)));
    cuda_error_chk(cudaMallocManaged(&A[0], n * m * sizeof(T)));
    break;
  }
  for (int i = 0; i < m; ++i) {
    A[i] = A[0] + i * n;
  }
}
