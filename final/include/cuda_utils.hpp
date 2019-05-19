#pragma once

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <stdio.h>

#ifdef __CUDACC__
#define cuda_error_chk(ans)                                                    \
  { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "cuda assert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}
#endif

static const char *cublas_get_error_string(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";

  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";

  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}

#define cublas_error_chk(ans)                                                  \
  { cublas_assert((ans), __FILE__, __LINE__); }
inline void cublas_assert(cublasStatus_t code, const char *file, int line) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublas assert: %s %s %d\n", cublas_get_error_string(code),
            file, line);
    exit(code);
  }
}

static const char *cusparse_get_error_string(cusparseStatus_t error) {
  switch (error) {
  case CUSPARSE_STATUS_SUCCESS:
    return "CUSPARSE_STATUS_SUCCESS";

  case CUSPARSE_STATUS_NOT_INITIALIZED:
    return "CUSPARSE_STATUS_NOT_INITIALIZED";

  case CUSPARSE_STATUS_ALLOC_FAILED:
    return "CUSPARSE_STATUS_ALLOC_FAILED";

  case CUSPARSE_STATUS_INVALID_VALUE:
    return "CUSPARSE_STATUS_INVALID_VALUE";

  case CUSPARSE_STATUS_ARCH_MISMATCH:
    return "CUSPARSE_STATUS_ARCH_MISMATCH";

  case CUSPARSE_STATUS_MAPPING_ERROR:
    return "CUSPARSE_STATUS_MAPPING_ERROR";

  case CUSPARSE_STATUS_EXECUTION_FAILED:
    return "CUSPARSE_STATUS_EXECUTION_FAILED";

  case CUSPARSE_STATUS_INTERNAL_ERROR:
    return "CUSPARSE_STATUS_INTERNAL_ERROR";

  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

  case CUSPARSE_STATUS_ZERO_PIVOT:
    return "CUSPARSE_STATUS_ZERO_PIVOT";
  }

  return "<unknown>";
}

#define cusparse_error_chk(ans)                                                \
  { cusparse_assert((ans), __FILE__, __LINE__); }
inline void cusparse_assert(cusparseStatus_t code, const char *file, int line) {
  if (code != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "cusparse assert: %s %s %d\n",
            cusparse_get_error_string(code), file, line);
    exit(code);
  }
}

namespace miniFE {
  void select_cuda_device(int mpi_rank);
}
