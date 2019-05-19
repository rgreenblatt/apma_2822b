#include <cuda.h>
#include <cuda_runtime.h>
#include "vector_uvm.hpp"

template <class T> T *UMAllocator<T>::allocate(size_t n) {
  T *ptr;
#ifdef USE_CUDA
  if (n > 0) {
    //        ptr = (T*) malloc(n*sizeof(T));
    //        cudaMemPrefetchAsync(ptr,n*sizeof(T),0,0);
    cudaMallocManaged(&ptr, n * sizeof(T));
    cudaMemAdvise(ptr, n * sizeof(T), cudaMemAdviseSetPreferredLocation, 0);
  } else {
    ptr = NULL;
  }
#else
  ptr = new T[n];
#endif
  return ptr;
}

template <class T> void UMAllocator<T>::deallocate(T *p, size_t) {
#ifdef USE_CUDA
  cudaFree(p);
#else
  delete p;
#endif
}

template struct UMAllocator<MINIFE_LOCAL_ORDINAL>;
template struct UMAllocator<MINIFE_GLOBAL_ORDINAL>;
template struct UMAllocator<MINIFE_SCALAR>;
