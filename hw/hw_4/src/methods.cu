#include "methods.h"
#include "utils.h"

void CRSMethodCPU::run() {
  // compute y = A*x
  // A is sparse operator stored in a CSR format

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

    // compute y = A*x
    // A is sparse operator stored in a CSR format
    const int J1 = IA[i];
    const int J2 = IA[i + 1];
    double sum = 0.0;
    for (int j = 0; j < (J2 - J1); j++)
      sum += AA[j + J1] * x[JA[j + J1]];
    y[i] = sum;
  }
}

void CRSMethodGPU::run() {
  int num_threads = 16;
  SpMv_gpu_thread_CRS<<<(Nrow + num_threads - 1) / num_threads, num_threads>>>(
      Nrow, AA, IA, JA, x, y);
  cuda_error_chk(cudaPeekAtLastError());
}

void ELLPACKMethodCPU::run() {
  // compute y = A*x
  // A is sparse operator stored in a ELLPACK format

  // Note, could be optimized using blocking
  for (int i = 0; i < Nrow; i++) {
    double sum = 0.0;
    for (int j = 0; j < maxnzr; j++) {
      int idx = i * maxnzr + j;
      sum += AS[idx] * x[JA[idx]];
    }
    y[i] = sum;
  }
}

__global__ void SpMv_gpu_thread_ELLPACK(int num_per_block_row, int Nrow,
                                        int maxnzr, double *AS, int *JA,
                                        double *x, double *y) {
  // compute y = A*x
  // A is sparse operator stored in a ELLPACK format

  int block_row = blockIdx.x * blockDim.x + threadIdx.x;
  int max_row = min((block_row + 1) * num_per_block_row, Nrow);

  for (int i = block_row * num_per_block_row; i < max_row; i++) {
    double sum = 0;
    for (int j = 0; j < maxnzr; j++) {
      int idx = i * maxnzr + j;
      sum += AS[idx] * x[JA[idx]];
    }
    y[i] = sum;
  }
}

void ELLPACKMethodGPU::run() {
  int num_threads = 256;
  int num_per_block_row = 1;

  int num_blocks_row = (Nrow + num_per_block_row - 1) / num_per_block_row;

  SpMv_gpu_thread_ELLPACK<<<(num_blocks_row + num_threads - 1) / num_threads,
                            num_threads>>>(num_per_block_row, Nrow, maxnzr, AS,
                                           JA, x, y);

  cuda_error_chk(cudaDeviceSynchronize());
  cuda_error_chk(cudaPeekAtLastError());
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
    : handle(handle), Nrow(Nrow), Ncol(Ncol), nnz(nnz), AA(AA), IA(IA), JA(JA),
      x(x), y(y) {
  cuda_sparse_error_chk(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
}
