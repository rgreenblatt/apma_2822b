#pragma once

#include <stdio.h>

/* #define cuda_sparse_error_chk(ans)                                             \ */
/*   { cuda_sparse_assert((ans), __FILE__, __LINE__); } */
/* inline void cuda_sparse_assert(cusparseStatus_t code, const char *file, */
/*                                int line) { */
/*   if (code != CUSPARSE_STATUS_SUCCESS) { */
/*     fprintf(stderr, "Cuda sparse error %d %s %d\n", code, file, line); */
/*     exit(code); */
/*   } */
/* } */

#define cuda_error_chk(ans)                                                    \
  { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}
