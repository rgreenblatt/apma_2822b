#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <string>

namespace chr = std::chrono;
using h_clock = chr::high_resolution_clock;

void SpMv_cpu(int Nrow, double *AA, int *IA, int *JA, double *x, double *y);

void SpMv_gpu(int Nrow, double *AA, int *IA, int *JA, double *x, double *y);

void time_function(int iterations,
                      void (*func)(int Nrow, double *AA, int *IA, int *JA,
                                   double *x, double *y), double* times,
                      int Nrow, double *AA, int *IA, int *JA, double *v,
                      double *rhs, bool is_cuda);
void print_result(double* times, int iterations, const char * name);

int main() {

  double *AA, *AA_copy_gpu, *AA_copy_cpu, *AA_managed;
  int *JA, *IA, *JA_copy_gpu, *JA_copy_cpu, *IA_copy_gpu, *IA_copy_cpu,
      *JA_managed, *IA_managed;
  double *v, *rhs, *v_copy_gpu, *v_copy_cpu, *rhs_copy_gpu, *rhs_copy_cpu,
      *v_managed, *rhs_managed;
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

  AA = new double[nnz];
  JA = new int[nnz];
  IA = new int[Nrow + 1];

  v = new double[Ncol];
  rhs = new double[Nrow];

  AA_copy_cpu = new double[nnz];
  JA_copy_cpu = new int[nnz];
  IA_copy_cpu = new int[Nrow + 1];

  v_copy_cpu = new double[Ncol];
  rhs_copy_cpu = new double[Nrow];

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

  cudaMalloc(&AA_copy_gpu, nnz * sizeof(double));
  cudaMalloc(&JA_copy_gpu, nnz * sizeof(double));
  cudaMalloc(&IA_copy_gpu, (Nrow + 1) * sizeof(double));
  cudaMalloc(&v_copy_gpu, Ncol * sizeof(double));
  cudaMalloc(&rhs_copy_gpu, Nrow * sizeof(double));

  cudaMemcpy(AA_copy_gpu, AA, nnz * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(JA_copy_gpu, JA, nnz * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(IA_copy_gpu, IA, (Nrow + 1) * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(v_copy_gpu, v, Ncol * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(rhs_copy_gpu, rhs, Nrow * sizeof(double), cudaMemcpyHostToDevice);

  cudaMallocManaged(&AA_managed, nnz * sizeof(double));
  cudaMallocManaged(&JA_managed, nnz * sizeof(double));
  cudaMallocManaged(&IA_managed, (Nrow + 1) * sizeof(double));
  cudaMallocManaged(&v_managed, Ncol * sizeof(double));
  cudaMallocManaged(&rhs_managed, Nrow * sizeof(double));

  cudaMemcpy(AA_managed, AA, nnz * sizeof(double), cudaMemcpyHostToHost);
  cudaMemcpy(JA_managed, JA, nnz * sizeof(double), cudaMemcpyHostToHost);
  cudaMemcpy(IA_managed, IA, (Nrow + 1) * sizeof(double), cudaMemcpyHostToHost);
  cudaMemcpy(v_managed, v, Ncol * sizeof(double), cudaMemcpyHostToHost);
  cudaMemcpy(rhs_managed, rhs, Nrow * sizeof(double), cudaMemcpyHostToHost);

  int iterations = 10;
  double cpu_times[iterations];
  double copy_gpu_times[iterations];
  double managed_cpu_times_before_gpu[iterations];
  double managed_gpu_times[iterations];
  double managed_cpu_times_after_gpu[iterations];
  double unused_times[iterations];

  // time different approaches
  time_function(iterations, SpMv_cpu, cpu_times, Nrow, AA, IA, JA, v, rhs, false);
  time_function(iterations, SpMv_gpu, copy_gpu_times, Nrow, AA_copy_gpu,
                IA_copy_gpu, JA_copy_gpu, v_copy_gpu, rhs_copy_gpu, true);

  // copy back to host
  // for unclear reasons, this must be done before using managed memory on the
  // cpu
  cudaMemcpy(AA_copy_gpu, AA_copy_cpu, nnz * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(JA_copy_gpu, JA_copy_cpu, nnz * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(IA_copy_gpu, IA_copy_cpu, (Nrow + 1) * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(v_copy_gpu, v_copy_cpu, Ncol * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(rhs_copy_gpu, rhs_copy_cpu, Nrow * sizeof(double), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  time_function(iterations, SpMv_cpu, managed_cpu_times_before_gpu, Nrow,
                AA_managed, IA_managed, JA_managed, v_managed, rhs_managed, false);
  time_function(iterations, SpMv_gpu, managed_gpu_times, Nrow, AA_managed,
                IA_managed, JA_managed, v_managed, rhs_managed, true);
  time_function(iterations, SpMv_gpu, managed_cpu_times_after_gpu, Nrow,
                AA_managed, IA_managed, JA_managed, v_managed, rhs_managed, false);
  time_function(iterations, SpMv_gpu, unused_times, Nrow, AA_managed,
                IA_managed, JA_managed, v_managed, rhs_managed, true);

  // verify that that the gpu function is correct
  for (i = 0; i < Nrow; i++) {
    assert(std::abs(rhs[i] - rhs_managed[i]) < TOL);
  }

  // print out results
  printf(" ======= Timings =======\n");
  print_result(cpu_times, iterations, "cpu");
  print_result(copy_gpu_times, iterations, "copy gpu");
  print_result(managed_cpu_times_before_gpu, iterations,
               "managed cpu before tranfer to the gpu");
  print_result(managed_gpu_times, iterations,
               "managed gpu");
  print_result(managed_cpu_times_after_gpu, iterations,
               "managed cpu after tranfer to the gpu");

  // trasnform sparse operator from the CSR to ELLPACK format

  delete[] AA;
  delete[] IA;
  delete[] JA;
  delete[] v;
  delete[] rhs;

  cudaFree(AA_copy_gpu);
  cudaFree(IA_copy_gpu);
  cudaFree(JA_copy_gpu);
  cudaFree(v_copy_gpu);
  cudaFree(rhs_copy_gpu);

  cudaFree(AA_managed);
  cudaFree(IA_managed);
  cudaFree(JA_managed);
  cudaFree(v_managed);
  cudaFree(rhs_managed);

  return 0;
}

void SpMv_cpu(int Nrow, double *AA, int *IA, int *JA, double *x, double *y) {

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

__global__ void SpMv_gpu_thread(double *AA, int *IA, int *JA, double *x,
                                double *y) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // compute y = A*x
  // A is sparse operator stored in a CSR format
  const int J1 = IA[i];
  const int J2 = IA[i + 1];
  double sum = 0.0;
  for (int j = 0; j < (J2 - J1); j++)
    sum += AA[j + J1] * x[JA[j + J1]];
  y[i] = sum;
}

void SpMv_gpu(int Nrow, double *AA, int *IA, int *JA, double *x, double *y) {
  SpMv_gpu_thread<<<(Nrow + 255) / 256, 256>>>(AA, IA, JA, x, y);
}
void time_function(int iterations, void (*func)(int Nrow, double *AA, int *IA, int *JA,
                                   double *x, double *y), double *times, int Nrow, double *AA,
                      int *IA, int *JA, double *v, double *rhs, bool is_cuda) {
  for (int i = 0; i < iterations; ++i) {
    auto t1 = h_clock::now();
    for (int j = 0; j < 10; j++) {
      (*func)(Nrow, AA, IA, JA, v, rhs);
    }
    if (is_cuda) {
      cudaDeviceSynchronize();
    }
    auto t2 = h_clock::now();
    double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();
    times[i] = time;
  }
}

void print_result(double *times, int iterations, const char * name) {
  printf("%s times: \nall times: ", name);
  for (int i = 0; i < iterations; i++) {
    printf("%e", times[i]);
    if (i != iterations - 1) {
      printf(",");
    }
  }
  assert(iterations > 2);
  double avg = 0.;
  for (int i = 2; i < iterations; i++) {
    avg += times[i];
  }
  avg /= (iterations - 2);

  printf("\naverage not including first two runs: %e\n\n", avg);
}
