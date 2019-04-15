#include <assert.h>
#include "methods.h"
#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include <string>


int main() {

  double TOL = 1.0e-10;
  int i, nnz, Nrow, Ncol;

  char fname[128], buf[256];
  FILE *SPfile;
  sprintf(fname, "SPoperator.23.0.dat");
  SPfile = fopen(fname, "r");

  if (SPfile == NULL)
    fprintf(stderr, "file %s  not found\n", fname);

  char *_ = fgets(buf, 128, SPfile);
  sscanf(buf, "%d %d %d", &nnz, &Nrow, &Ncol);

  fprintf(stdout, "nnz=%d Nrow=%d Ncol=%d nnz per row = %g\n", nnz, Nrow, Ncol,
          (double)nnz / Nrow);

  double *AA, *v, *v_copy_gpu, *v_managed, *rhs, *rhs_copy_gpu,
      *rhs_copy_cpu_mine, *rhs_copy_cpu_cuda_sparse, *rhs_managed, *true_rhs;
  int *IA, *JA;

  allocate_vector(AA, nnz, MemoryType::Host);
  allocate_vector(JA, nnz, MemoryType::Host);
  allocate_vector(IA, Nrow + 1, MemoryType::Host);
  allocate_vector(v, Ncol, MemoryType::Host);
  allocate_vector(rhs, Nrow, MemoryType::Host);

  allocate_vector(rhs_copy_cpu_mine, Nrow, MemoryType::Host);
  allocate_vector(rhs_copy_cpu_cuda_sparse, Nrow, MemoryType::Host);

  allocate_vector(true_rhs, Nrow, MemoryType::Host);

  allocate_vector(v_copy_gpu, Ncol, MemoryType::Device);
  allocate_vector(rhs_copy_gpu, Nrow, MemoryType::Device);

  allocate_vector(v_managed, Ncol, MemoryType::Unified);
  allocate_vector(rhs_managed, Nrow, MemoryType::Unified);

  for (i = 0; i <= Nrow; i++) {
    char *_ = fgets(buf, 128, SPfile);
    sscanf(buf, "%d\n", &IA[i]);
  }
  for (i = 0; i < nnz; i++) {
    char *_ = fgets(buf, 128, SPfile);
    sscanf(buf, "%d %lf\n", &JA[i], &AA[i]);
  }
  fclose(SPfile);

  fprintf(stdout, "done reading data\n");

  // initialize "v" and "rhs"
  for (i = 0; i < Ncol; i++)
    v[i] = 1.0;
  for (i = 0; i < Nrow; i++)
    rhs[i] = 1.0;

  cuda_error_chk(
      cudaMemcpy(v_copy_gpu, v, Ncol * sizeof(double), cudaMemcpyHostToDevice));
  cuda_error_chk(cudaMemcpy(rhs_copy_gpu, rhs, Nrow * sizeof(double),
                            cudaMemcpyHostToDevice));

  cuda_error_chk(
      cudaMemcpy(v_managed, v, Ncol * sizeof(double), cudaMemcpyHostToHost));
  cuda_error_chk(cudaMemcpy(rhs_managed, rhs, Nrow * sizeof(double),
                            cudaMemcpyHostToHost));

  int maxnzr = -1;
  double average_nzr = 0.;
  for (int i = 0; i < Nrow; i++) {
    const int candidate = IA[i + 1] - IA[i];
    average_nzr += candidate;
    if (candidate > maxnzr) {
      maxnzr = candidate;
    }
  }
  average_nzr /= Nrow;

  printf("average nzr count is: %f, max nzr count is: %d", average_nzr, maxnzr);
  {
    double *AA_copy_gpu, *AA_managed;
    int *JA_copy_gpu, *IA_copy_gpu, *JA_managed, *IA_managed;

    allocate_vector(AA_copy_gpu, nnz, MemoryType::Device);
    allocate_vector(JA_copy_gpu, nnz, MemoryType::Device);
    allocate_vector(IA_copy_gpu, Nrow + 1, MemoryType::Device);

    cuda_error_chk(cudaMemcpy(AA_copy_gpu, AA, nnz * sizeof(double),
                              cudaMemcpyHostToDevice));
    cuda_error_chk(
        cudaMemcpy(JA_copy_gpu, JA, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cuda_error_chk(cudaMemcpy(IA_copy_gpu, IA, (Nrow + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));

    allocate_vector(AA_managed, nnz, MemoryType::Unified);
    allocate_vector(JA_managed, nnz, MemoryType::Unified);
    allocate_vector(IA_managed, (Nrow + 1), MemoryType::Unified);

    cuda_error_chk(
        cudaMemcpy(AA_managed, AA, nnz * sizeof(double), cudaMemcpyHostToHost));
    cuda_error_chk(
        cudaMemcpy(JA_managed, JA, nnz * sizeof(int), cudaMemcpyHostToHost));
    cuda_error_chk(cudaMemcpy(IA_managed, IA, (Nrow + 1) * sizeof(int),
                              cudaMemcpyHostToHost));

    cuda_error_chk(cudaDeviceSynchronize());

    int iterations = 10;
    double cpu_times[iterations];
    double gpu_times[iterations];
    double cpu_managed_times_before_gpu[iterations];
    double cpu_managed_times_after_gpu[iterations];
    double gpu_managed_times[iterations];

    CRSMethodCPU cpu(Nrow, AA, IA, JA, v, rhs);
    CRSMethodGPU gpu(Nrow, AA_copy_gpu, IA_copy_gpu, JA_copy_gpu, v_copy_gpu,
                     rhs_copy_gpu);

    time_function(iterations, cpu, cpu_times, false);
    std::memcpy(true_rhs, rhs, sizeof(double) * Nrow);
    time_function(iterations, gpu, gpu_times, true);

    // copy back to host
    cuda_error_chk(cudaMemcpy(rhs_copy_cpu_mine, rhs_copy_gpu,
                              Nrow * sizeof(double), cudaMemcpyDeviceToHost));
    cuda_error_chk(cudaDeviceSynchronize());

    CRSMethodCPU cpu_managed(Nrow, AA_managed, IA_managed, JA_managed,
                             v_managed, rhs_managed);
    CRSMethodGPU gpu_managed(Nrow, AA_managed, IA_managed, JA_managed,
                             v_managed, rhs_managed);

    time_function(iterations, cpu_managed, cpu_managed_times_before_gpu, false);
    time_function(iterations, gpu_managed, gpu_managed_times, true);
    time_function(iterations, cpu_managed, cpu_managed_times_after_gpu, false);

    cusparseHandle_t handle;
    cuda_sparse_error_chk(cusparseCreate(&handle));
    CudaSparse cuda_sparse(handle, Nrow, Ncol, nnz, AA_copy_gpu, IA_copy_gpu,
                           JA_copy_gpu, v_copy_gpu, rhs_copy_gpu);

    double cuda_sparse_times[iterations];
    time_function(iterations, cuda_sparse, cuda_sparse_times, true);

    // copy back to host
    cuda_error_chk(cudaMemcpy(rhs_copy_cpu_cuda_sparse, rhs_copy_gpu,
                              Nrow * sizeof(double), cudaMemcpyDeviceToHost));
    cuda_error_chk(cudaDeviceSynchronize());

    // print out results
    printf("\n======= Timings Cublas =======\n");
    print_result(gpu_times, iterations, "gpu");

    // verify correctness
    for (int i = 0; i < Nrow; i++) {
      assert(std::abs(true_rhs[i] - rhs_managed[i]) < TOL);
      assert(std::abs(true_rhs[i] - rhs_copy_cpu_mine[i]) < TOL);
      assert(std::abs(true_rhs[i] - rhs_copy_cpu_cuda_sparse[i]) < TOL);
    }

    // print out results
    printf("\n======= Timings CRS =======\n");
    print_result(cpu_times, iterations, "cpu");
    print_result(gpu_times, iterations, "gpu");
    print_result(cpu_managed_times_before_gpu, iterations,
                 "cpu managed before tranfer to the gpu");
    print_result(gpu_managed_times, iterations, "gpu managed");
    print_result(cpu_managed_times_before_gpu, iterations,
                 "cpu managed after tranfer to the gpu");
    print_result(cuda_sparse_times, iterations, "cuda sparse library");
    cuda_error_chk(cudaFree(AA_copy_gpu));
    cuda_error_chk(cudaFree(IA_copy_gpu));
    cuda_error_chk(cudaFree(JA_copy_gpu));

    cuda_error_chk(cudaFree(AA_managed));
    cuda_error_chk(cudaFree(IA_managed));
    cuda_error_chk(cudaFree(JA_managed));
  }

  // reset "v" and "rhs"
  for (int i = 0; i < Ncol; i++)
    v[i] = 1.0;
  for (i = 0; i < Nrow; i++)
    rhs[i] = 1.0;

  printf("\n");
  {

    assert(maxnzr >= 0);

    // ELLPACK
    double **AS, *AS_copy_gpu, **AS_managed;
    int **JA_E, *JA_E_copy_gpu, **JA_E_managed;
    int *row_lengths, *row_lengths_copy_gpu, *row_lengths_managed;

    allocate_matrix(AS, Nrow, maxnzr, MemoryType::Host);
    allocate_matrix(JA_E, Nrow, maxnzr, MemoryType::Host);
    allocate_vector(row_lengths, Nrow, MemoryType::Host);

    size_t pitch_AS =
        allocate_matrix_device(AS_copy_gpu, Nrow, maxnzr);
    size_t pitch_JA_E =
        allocate_matrix_device(JA_E_copy_gpu, Nrow, maxnzr);
    allocate_vector(row_lengths_copy_gpu, Nrow, MemoryType::Device);

    allocate_matrix(AS_managed, Nrow, maxnzr, MemoryType::Unified);
    allocate_matrix(JA_E_managed, Nrow, maxnzr, MemoryType::Unified);
    allocate_vector(row_lengths_managed, Nrow, MemoryType::Unified);

    // transform sparse operator from the CSR to ELLPACK format
    for (int i = 0; i < Nrow; i++) {
      const int J1 = IA[i];
      const int J2 = IA[i + 1];
      row_lengths[i] = J2 - J1;
      for (int j = 0; j < maxnzr; j++) {
        if (j < J2 - J1) {
          AS[i][j] = AA[J1 + j];
          JA_E[i][j] = JA[j + J1];
        } else {
          AS[i][j] = 0.;
          JA_E[i][j] = JA[J2 - 1];
        }
      }
    }

    cuda_error_chk(cudaMemcpy2D(AS_copy_gpu, pitch_AS, AS[0], maxnzr * sizeof(double),
                                maxnzr * sizeof(double), Nrow,
                                cudaMemcpyHostToDevice));
    cuda_error_chk(
        cudaMemcpy2D(JA_E_copy_gpu, pitch_JA_E, JA_E[0], maxnzr * sizeof(int),
                     maxnzr * sizeof(int), Nrow, cudaMemcpyHostToDevice));

    cuda_error_chk(cudaMemcpy(row_lengths_copy_gpu, row_lengths,
                              Nrow * sizeof(int),
                              cudaMemcpyHostToDevice));

    cuda_error_chk(cudaMemcpy(AS_managed[0], AS[0], Nrow * maxnzr * sizeof(double),
                              cudaMemcpyHostToHost));
    cuda_error_chk(cudaMemcpy(JA_E_managed[0], JA_E[0], Nrow * maxnzr * sizeof(int),
                              cudaMemcpyHostToHost));
    cuda_error_chk(cudaMemcpy(row_lengths_managed, row_lengths,
                              Nrow * sizeof(int),
                              cudaMemcpyHostToHost));

    cuda_error_chk(cudaDeviceSynchronize());

    int iterations = 10;
    double cpu_times[iterations];
    double gpu_times[iterations];
    double cpu_managed_times_before_gpu[iterations];
    double cpu_managed_times_after_gpu[iterations];
    double gpu_managed_times[iterations];


    /***/
    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;


    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = v_copy_gpu;
    resDesc.res.linear.sizeInBytes = Nrow * sizeof(double);
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.desc.y = 32;

    cudaTextureObject_t texObject;
    cuda_error_chk(cudaCreateTextureObject(&texObject, &resDesc, &td, NULL));

    ELLPACKMethodCPU cpu(Nrow, maxnzr, row_lengths, AS, JA_E, v, rhs);
    ELLPACKMethodGPU gpu(Nrow, maxnzr, row_lengths_copy_gpu, AS_copy_gpu, pitch_AS,
                         JA_E_copy_gpu, pitch_JA_E, texObject, rhs_copy_gpu);



    time_function(iterations, cpu, cpu_times, false);
    time_function(iterations, gpu, gpu_times, true);

    // copy back to host
    cuda_error_chk(cudaMemcpy(rhs_copy_cpu_mine, rhs_copy_gpu,
                              Nrow * sizeof(double), cudaMemcpyDeviceToHost));
    cuda_error_chk(cudaDeviceSynchronize());

    ELLPACKMethodCPU cpu_managed(Nrow, maxnzr, row_lengths_managed, AS_managed,
                                 JA_E_managed, v_managed, rhs_managed);
    ELLPACKMethodGPUManaged gpu_managed(Nrow, maxnzr, row_lengths_managed, AS_managed,
                                 JA_E_managed, v_managed, rhs_managed);

    time_function(iterations, cpu_managed, cpu_managed_times_before_gpu, false);
    time_function(iterations, gpu_managed, gpu_managed_times, true);
    time_function(iterations, cpu_managed, cpu_managed_times_after_gpu, false);

    // this must be last for checking that the gpu computation is correct
    gpu_managed.run();
    cuda_error_chk(cudaDeviceSynchronize());

    // verify correctness
    for (int i = 0; i < Nrow; i++) {
      assert(std::abs(true_rhs[i] - rhs[i]) < TOL);
      assert(std::abs(true_rhs[i] - rhs_managed[i]) < TOL);
      assert(std::abs(true_rhs[i] - rhs_copy_cpu_mine[i]) < TOL);
    }

    // print out results
    printf("\n======= Timings ELLPACK =======\n");
    print_result(cpu_times, iterations, "cpu");
    print_result(gpu_times, iterations, "gpu");
    print_result(cpu_managed_times_before_gpu, iterations,
                 "cpu managed before tranfer to the gpu");
    print_result(gpu_managed_times, iterations, "gpu managed");
    print_result(cpu_managed_times_before_gpu, iterations,
                 "cpu managed after tranfer to the gpu");

    delete[] AS;
    delete[] JA_E;

    cuda_error_chk(cudaFree(AS_copy_gpu));
    cuda_error_chk(cudaFree(JA_E_copy_gpu));

    cuda_error_chk(cudaFree(AS_managed));
    cuda_error_chk(cudaFree(JA_E_managed));
  }

  // reset "v" and "rhs"
  for (int i = 0; i < Ncol; i++)
    v[i] = 1.0;
  for (i = 0; i < Nrow; i++)
    rhs[i] = 1.0;

  printf("\n");
  {}

  delete[] AA;
  delete[] IA;
  delete[] JA;
  delete[] v;
  delete[] rhs;

  cuda_error_chk(cudaFree(v_copy_gpu));
  cuda_error_chk(cudaFree(rhs_copy_gpu));

  cuda_error_chk(cudaFree(v_managed));
  cuda_error_chk(cudaFree(rhs_managed));

  return 0;
}
