
#ifndef  VECTOR_UVM_H 
#define VECTOR_UVM_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


template <class T>
struct UMAllocator {
  typedef T value_type;
  UMAllocator() {}
  template <class U> UMAllocator(const UMAllocator<U>& other);

  T* allocate(std::size_t n)
  {
    T* ptr;
#ifdef USE_UVM
      if (n>0){
//        ptr = (T*) malloc(n*sizeof(T));
//        cudaMemPrefetchAsync(ptr,n*sizeof(T),0,0);
      cudaMallocManaged(&ptr, n*sizeof(T));
      cudaMemAdvise(ptr,n*sizeof(T),cudaMemAdviseSetPreferredLocation,0);
      }
      else {
        ptr = NULL;
      }
#else
    ptr = (T*) malloc(n*sizeof(T));
#endif
    return ptr;
  }

  void deallocate(T* p, std::size_t n)
  {
#ifdef USE_UVM
    cudaFree(p);
//      if (p!= NULL) free(p);
#else
    free(p);
#endif
  }
};

template <class T, class U>
bool operator==(const UMAllocator<T>&, const UMAllocator<U>&);
template <class T, class U>
bool operator!=(const UMAllocator<T>&, const UMAllocator<U>&);


#endif

