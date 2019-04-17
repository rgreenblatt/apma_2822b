#include "methods.h"
#include "utils.h"
#include <algorithm>

void CRSMethodCPU::run() {
  for (int i = 0; i < Nrow; i++) {
    const int J1 = IA[i];
    const int J2 = IA[i + 1];
    double sum = 0.0;
    for (int j = 0; j < (J2 - J1); j++)
      sum += AA[j + J1] * x[JA[j + J1]];
    y[i] = sum;
  }
}

__global__ void SpMv_gpu_thread_CRS(int Nrow, double *AA, int *IA, int *JA,
                                    double *x, double *y) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < Nrow) {

    const int J1 = IA[i];
    const int J2 = IA[i + 1];
    double sum = 0.0;
    for (int j = 0; j < (J2 - J1); j++)
      sum += AA[j + J1] * x[JA[j + J1]];
    y[i] = sum;
  }
}

void CRSMethodGPU::run() {
  int num_threads = 64;
  SpMv_gpu_thread_CRS<<<(Nrow + num_threads - 1) / num_threads, num_threads>>>(
      Nrow, AA, IA, JA, x, y);
}

void CudaSparse::run() {
  double alpha = 1.;
  double beta = 0.;
  cuda_sparse_error_chk(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       Nrow, Ncol, nnz, &alpha, descr, AA, IA,
                                       JA, x, &beta, y));
}

CudaSparse::CudaSparse(cusparseHandle_t handle, int Nrow, int Ncol, int nnz,
                       double *AA, int *IA, int *JA, double *x, double *y)
    : CRSMethod(Nrow, AA, IA, JA, x, y), handle(handle), Ncol(Ncol), nnz(nnz) {
  cuda_sparse_error_chk(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
}

void ELLPACKMethodCPU::run() {
  const int unroll_num = 4;
  #pragma omp parallel for
  for (int i = 0; i <  (Nrow / unroll_num) * unroll_num ; i += unroll_num) {
    double sum[unroll_num] = {0};

    int unroll_maxnzr = row_lengths[i];
    for (int k = 1; k < unroll_num; ++k) {
      unroll_maxnzr = std::max(unroll_maxnzr, row_lengths[i + k]);
    }
    for (int j = 0; j < unroll_maxnzr; j++) {
    #pragma unroll unroll_num
      for (int k = 0; k < unroll_num; k++) {
        sum[k] += AS[j][i + k] * x[JA[j][i + k]];
      }
    }
    #pragma unroll unroll_num
    for (int k = 0; k < unroll_num; k++) {
      y[i + k] = sum[k];
    }
  }
  for (int i = Nrow - Nrow % unroll_num; i < Nrow; i++) {
    double sum = 0;
    for (int j = 0; j < row_lengths[i]; j++) {
      sum += AS[j][i] * x[JA[j][i]];
    }
    y[i] = sum;
  }
}

static __inline__ __device__ double fetch_double(uint2 p) {
  return __hiloint2double(p.y, p.x);
}

__device__ double SpMv_gpu_thread_ELLPACK_row(int row_length, double **AS,
                                              int **JA, cudaTextureObject_t x,
                                              int row) {
  double sum = 0;
  for (int j = 0; j < row_length; j++) {
    sum += AS[j][row] * fetch_double(tex1Dfetch<uint2>(x, JA[j][row]));

  }
  return sum;
}

__global__ void SpMv_gpu_thread_ELLPACK(int Nrow, int maxnzr, int *row_lengths,
                                        double **AS, int **JA,
                                        cudaTextureObject_t x, double *y) {
  // compute y = A*x
  // A is sparse operator stored in a ELLPACK format

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < Nrow) {
    y[row] =
        SpMv_gpu_thread_ELLPACK_row(row_lengths[row], AS, JA, x, row);
  }
}

void ELLPACKMethodGPU::run() {
  int num_threads = 64;

  SpMv_gpu_thread_ELLPACK<<<(Nrow + num_threads - 1) / num_threads,
                            num_threads>>>(Nrow, maxnzr, row_lengths, AS, JA, x,
                                           y);
}
