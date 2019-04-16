#include "methods.h"
#include "utils.h"

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
  const int unroll = 4;
  for (int i = 0; i < Nrow; i+=unroll) {
    double sum[unroll] = {0};
    for (int j = 0; j < maxnzr; j++) {
      #pragma unroll
      for (int k = 0; k < unroll; k++) {
        if (i + k < Nrow) {
          sum[k] += AS[i + k][j] * x[JA[i + k][j]];
        }
      }
    }
    for (int k = 0; k < unroll; k++) {
      if (i + k < Nrow) {
        y[i + k] = sum[k];
      }
    }
  }
}

static __inline__ __device__ double fetch_double(uint2 p) {
  return __hiloint2double(p.y, p.x);
}

__device__ double SpMv_gpu_thread_ELLPACK_row(int row_length, double *AS,
                                              int *JA, cudaTextureObject_t x) {
  double sum = 0;
  for (int j = 0; j < row_length; j++) {
    sum += AS[j] * fetch_double(tex1Dfetch<uint2>(x, JA[j]));
  }
  return sum;
}

__device__ double SpMv_gpu_thread_ELLPACK_row_managed(int row_length,
                                                      double *AS, int *JA,
                                                      double *x) {
  double sum = 0;
  for (int j = 0; j < row_length; j++) {
    sum += AS[j] * x[JA[j]];
  }
  return sum;
}

__global__ void SpMv_gpu_thread_ELLPACK_managed(int Nrow, int maxnzr,
                                                int *row_lengths, double **AS,
                                                int **JA, double *x,
                                                double *y) {
  // compute y = A*x
  // A is sparse operator stored in a ELLPACK format

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < Nrow) {
    y[row] = SpMv_gpu_thread_ELLPACK_row_managed(row_lengths[row], AS[row],
                                                 JA[row], x);
  }
}

__global__ void SpMv_gpu_thread_ELLPACK(int Nrow, int maxnzr, int *row_lengths,
                                        double *AS, int *JA,
                                        cudaTextureObject_t x, double *y,
                                        size_t pitch_AS, size_t pitch_JA) {
  // compute y = A*x
  // A is sparse operator stored in a ELLPACK format

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < Nrow) {
    double *row_AS = (double *)((char *)AS + row * pitch_AS);
    int *row_JA = (int *)((char *)JA + row * pitch_JA);
    y[row] = SpMv_gpu_thread_ELLPACK_row(row_lengths[row], row_AS, row_JA, x);
  }
}

void ELLPACKMethodGPU::run() {
  int num_threads = 64;

  SpMv_gpu_thread_ELLPACK<<<(Nrow + num_threads - 1) / num_threads,
                            num_threads>>>(Nrow, maxnzr, row_lengths, AS, JA, x,
                                           y, pitch_AS, pitch_JA);

}

void ELLPACKMethodGPUManaged::run() {
  int num_threads = 64;

  SpMv_gpu_thread_ELLPACK_managed<<<(Nrow + num_threads - 1) / num_threads,
                                    num_threads>>>(Nrow, maxnzr, row_lengths,
                                                   AS, JA, x, y);

}
