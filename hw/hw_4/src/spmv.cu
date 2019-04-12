#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <stdio.h>
#include <string>
#include <cstring>

namespace chr = std::chrono;
using h_clock = chr::high_resolution_clock;

enum class MemoryType { Host, Device, Unified };

class SpMvMethod {
public:
  virtual void run() const = 0;
};

class CRSMethod : public SpMvMethod {
protected:
  int Nrow;
  double *AA;
  int *IA;
  int *JA;
  double *x;
  double *y;

public:
  CRSMethod(int Nrow, double *AA, int *IA, int *JA, double *x, double *y)
      : Nrow(Nrow), AA(AA), IA(IA), JA(JA), x(x), y(y) {}
};

class CRSMethodCPU : public CRSMethod {
public:
  void run() const;
  CRSMethodCPU(int Nrow, double *AA, int *IA, int *JA, double *x, double *y)
      : CRSMethod(Nrow, AA, IA, JA, x,
                  y) {}
};

class CRSMethodGPU : public CRSMethod {
public:
  void run() const ;
  CRSMethodGPU(int Nrow, double *AA, int *IA, int *JA, double *x, double *y)
      : CRSMethod(Nrow, AA, IA, JA, x,
                  y) {}
};

class ELLPACKMethod : public SpMvMethod {
protected:
  int Nrow;
  int maxnzr;
  double *AS;
  int *JA;
  double *x;
  double *y;

public:
  ELLPACKMethod(int Nrow, int maxnzr, double *AS, int *JA,
                double *x, double *y)
      : Nrow(Nrow), maxnzr(maxnzr), AS(AS), JA(JA), x(x),
        y(y) {}
};

class ELLPACKMethodCPU : public ELLPACKMethod {public:
public:
  void run() const;
  ELLPACKMethodCPU(int Nrow, int maxnzr, double *AS, int *JA,
                   double *x, double *y)
      : ELLPACKMethod(Nrow, maxnzr, AS, JA, x, y) {}
};

class ELLPACKMethodGPU : public ELLPACKMethod {
public:
  void run() const;
  ELLPACKMethodGPU(int Nrow, int maxnzr, double *AS, int *JA,
                   double *x, double *y)
      : ELLPACKMethod(Nrow, maxnzr, AS, JA, x, y) {}
};

#define cudaErrchk(ans)                                                        \
  { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
void SpMv_cpu(int Nrow, double *AA, int *IA, int *JA, double *x, double *y);

void SpMv_gpu(int Nrow, double *AA, int *IA, int *JA, double *x, double *y);

void time_function(int iterations, const SpMvMethod &method, double *times,
                   bool is_cuda);

void print_result(double *times, int iterations, const char *name);


template <class T> void allocate_vector(T *&A, int n, MemoryType memory_type);
template <class T> void allocate_matrix(T **&A, int n, int m, MemoryType memory_type);

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

  double *AA, *v, *v_copy_gpu, *v_managed, *rhs, *rhs_copy_gpu, *rhs_copy_cpu,
      *rhs_managed, *true_rhs;
  int *IA, *JA;

  allocate_vector(AA, nnz, MemoryType::Host);
  allocate_vector(JA, nnz, MemoryType::Host);
  allocate_vector(IA, Nrow + 1, MemoryType::Host);
  allocate_vector(v, Ncol, MemoryType::Host);
  allocate_vector(rhs, Nrow, MemoryType::Host);

  allocate_vector(rhs_copy_cpu, Nrow, MemoryType::Host);

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

  cudaErrchk(
      cudaMemcpy(v_copy_gpu, v, Ncol * sizeof(double), cudaMemcpyHostToDevice));
  cudaErrchk(cudaMemcpy(rhs_copy_gpu, rhs, Nrow * sizeof(double),
                        cudaMemcpyHostToDevice));

  cudaErrchk(
      cudaMemcpy(v_managed, v, Ncol * sizeof(double), cudaMemcpyHostToHost));
  cudaErrchk(cudaMemcpy(rhs_managed, rhs, Nrow * sizeof(double),
                        cudaMemcpyHostToHost));

  {
    double *AA_copy_gpu, *AA_managed;
    int *JA_copy_gpu, *IA_copy_gpu, *JA_managed, *IA_managed;

    allocate_vector(AA_copy_gpu, nnz, MemoryType::Device);
    allocate_vector(JA_copy_gpu, nnz, MemoryType::Device);
    allocate_vector(IA_copy_gpu, Nrow + 1, MemoryType::Device);

    cudaErrchk(cudaMemcpy(AA_copy_gpu, AA, nnz * sizeof(double),
                          cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(JA_copy_gpu, JA, nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(IA_copy_gpu, IA, (Nrow + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    allocate_vector(AA_managed, nnz, MemoryType::Unified);
    allocate_vector(JA_managed, nnz, MemoryType::Unified);
    allocate_vector(IA_managed, (Nrow + 1), MemoryType::Unified);

    cudaErrchk(
        cudaMemcpy(AA_managed, AA, nnz * sizeof(double), cudaMemcpyHostToHost));
    cudaErrchk(
        cudaMemcpy(JA_managed, JA, nnz * sizeof(int), cudaMemcpyHostToHost));
    cudaErrchk(cudaMemcpy(IA_managed, IA, (Nrow + 1) * sizeof(int),
                          cudaMemcpyHostToHost));

    cudaErrchk(cudaDeviceSynchronize());

    int iterations = 10;
    double cpu_times[iterations];
    double gpu_times[iterations];
    double cpu_managed_times_before_gpu[iterations];
    double cpu_managed_times_after_gpu[iterations];
    double gpu_managed_times[iterations];

    CRSMethodCPU cpu(Nrow, AA, IA, JA, v, rhs);
    CRSMethodGPU gpu(Nrow, AA_copy_gpu, IA_copy_gpu, JA_copy_gpu,
                         v_copy_gpu, rhs_copy_gpu);

    time_function(iterations, cpu, cpu_times, false);
    std::memcpy(true_rhs, rhs, sizeof(double) * Nrow);
    time_function(iterations, gpu, gpu_times, true);

    // copy back to host
    cudaErrchk(cudaMemcpy(rhs_copy_cpu, rhs_copy_gpu, Nrow * sizeof(double),
                          cudaMemcpyDeviceToHost));
    cudaErrchk(cudaDeviceSynchronize());

    CRSMethodCPU cpu_managed(Nrow, AA_managed, IA_managed, JA_managed,
                                 v_managed, rhs_managed);
    CRSMethodGPU gpu_managed(Nrow, AA_managed, IA_managed, JA_managed,
                                 v_managed, rhs_managed);

    time_function(iterations, cpu_managed, cpu_managed_times_before_gpu,
                  false);
    time_function(iterations, gpu_managed, gpu_managed_times, true);
    time_function(iterations, cpu_managed, cpu_managed_times_after_gpu,
                  false);

    // verify correctness
    for (int i = 0; i < Nrow; i++) {
      assert(std::abs(true_rhs[i] - rhs_managed[i]) < TOL);
      assert(std::abs(true_rhs[i] - rhs_copy_cpu[i]) < TOL);
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
    cudaErrchk(cudaFree(AA_copy_gpu));
    cudaErrchk(cudaFree(IA_copy_gpu));
    cudaErrchk(cudaFree(JA_copy_gpu));

    cudaErrchk(cudaFree(AA_managed));
    cudaErrchk(cudaFree(IA_managed));
    cudaErrchk(cudaFree(JA_managed));
  }

  // initialize "v" and "rhs"
  for (int i = 0; i < Ncol; i++)
    v[i] = 1.0;
  for (i = 0; i < Nrow; i++)
    rhs[i] = 1.0;

  printf("\n");
  {
    int maxnzr = -1;
    for (int i = 0; i < Nrow; i++) {
      const int candidate = IA[i + 1] - IA[i];
      if (candidate > maxnzr) {
        maxnzr = candidate;
      }
    }

    assert(maxnzr >= 0);

    // ELLPACK
    double *AS, *AS_copy_gpu, *AS_managed;
    int *JA_E, *JA_E_copy_gpu, *JA_E_managed;

    allocate_vector(AS, Nrow * maxnzr, MemoryType::Host);
    allocate_vector(JA_E, Nrow * maxnzr, MemoryType::Host);

    allocate_vector(AS_copy_gpu, Nrow * maxnzr, MemoryType::Device);
    allocate_vector(JA_E_copy_gpu, Nrow * maxnzr, MemoryType::Device);

    allocate_vector(JA_E_managed, Nrow * maxnzr, MemoryType::Unified);
    allocate_vector(AS_managed, Nrow * maxnzr, MemoryType::Unified);

    //transform sparse operator from the CSR to ELLPACK format
    for (int i = 0; i < Nrow; i++) {
      const int J1 = IA[i];
      const int J2 = IA[i + 1];
      for (int j = 0; j < maxnzr; j++) {
        int idx = i * maxnzr + j;
        if (j < J2 - J1) {
          AS[idx] = AA[J1 + j];
          JA_E[idx] = JA[j + J1];
        } else {
          AS[idx] = 0.;
          JA_E[idx] = JA[J2 - 1];
        }
      }
    }

    cudaErrchk(cudaMemcpy(AS_copy_gpu, AS, Nrow * maxnzr * sizeof(double),
                          cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(JA_E_copy_gpu, JA_E, Nrow * maxnzr * sizeof(int),
                          cudaMemcpyHostToDevice));

    cudaErrchk(cudaMemcpy(AS_managed, AS, Nrow * maxnzr * sizeof(double),
                          cudaMemcpyHostToHost));
    cudaErrchk(cudaMemcpy(JA_E_managed, JA_E, Nrow * maxnzr * sizeof(int),
                          cudaMemcpyHostToHost));

    cudaErrchk(cudaDeviceSynchronize());

    int iterations = 10;
    double cpu_times[iterations];
    double gpu_times[iterations];
    double cpu_managed_times_before_gpu[iterations];
    double cpu_managed_times_after_gpu[iterations];
    double gpu_managed_times[iterations];

    ELLPACKMethodCPU cpu(Nrow, maxnzr, AS, JA_E, v, rhs);
    ELLPACKMethodGPU gpu(Nrow, maxnzr, AS_copy_gpu, JA_E_copy_gpu, v_copy_gpu,
        rhs_copy_gpu);

    time_function(iterations, cpu, cpu_times, false);
    time_function(iterations, gpu, gpu_times, true);

    // copy back to host
    cudaErrchk(cudaMemcpy(rhs_copy_cpu, rhs_copy_gpu, Nrow * sizeof(double),
                          cudaMemcpyDeviceToHost));
    cudaErrchk(cudaDeviceSynchronize());

    ELLPACKMethodCPU cpu_managed(Nrow, maxnzr, AS_managed, JA_E_managed,
                                 v_managed, rhs_managed);
    ELLPACKMethodGPU gpu_managed(Nrow, maxnzr, AS_managed, JA_E_managed,
                                 v_managed, rhs_managed);

    time_function(iterations, cpu_managed, cpu_managed_times_before_gpu,
                  false);
    time_function(iterations, gpu_managed, gpu_managed_times, true);
    time_function(iterations, cpu_managed, cpu_managed_times_after_gpu,
                  false);

    // this must be last for checking that the gpu computation is correct
    gpu_managed.run();
    cudaErrchk(cudaDeviceSynchronize());


    // verify correctness
    for (int i = 0; i < Nrow; i++) {
      assert(std::abs(true_rhs[i] - rhs[i]) < TOL);
      assert(std::abs(true_rhs[i] - rhs_managed[i]) < TOL);
      assert(std::abs(true_rhs[i] - rhs_copy_cpu[i]) < TOL);
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

    cudaErrchk(cudaFree(AS_copy_gpu));
    cudaErrchk(cudaFree(JA_E_copy_gpu));

    cudaErrchk(cudaFree(AS_managed));
    cudaErrchk(cudaFree(JA_E_managed));
  }

  delete[] AA;
  delete[] IA;
  delete[] JA;
  delete[] v;
  delete[] rhs;

  cudaErrchk(cudaFree(v_copy_gpu));
  cudaErrchk(cudaFree(rhs_copy_gpu));

  cudaErrchk(cudaFree(v_managed));
  cudaErrchk(cudaFree(rhs_managed));

  return 0;
}

void CRSMethodCPU::run() const {
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

void CRSMethodGPU::run() const {
  int num_threads = 16;
  SpMv_gpu_thread_CRS<<<(Nrow + num_threads - 1) / num_threads, num_threads>>>(
      Nrow, AA, IA, JA, x, y);
  cudaErrchk(cudaPeekAtLastError());
}

void ELLPACKMethodCPU::run() const {
  // compute y = A*x
  // A is sparse operator stored in a ELLPACK format

  //Note, could be optimized using blocking
  for (int i = 0; i < Nrow; i++) {
    double sum = 0.0;
    for (int j = 0; j < maxnzr; j++) {
      int idx = i * maxnzr  + j;
      sum += AS[idx] * x[JA[idx]];
    }
    y[i] = sum;
  }
}

__global__ void SpMv_gpu_thread_ELLPACK(int num_per_block_row, int Nrow,
                                        int num_per_block_maxnzr,
                                        int num_blocks_maxnzr, int maxnzr,
                                        double *AS, int *JA, double *x,
                                        double *y) {
  // compute y = A*x
  // A is sparse operator stored in a ELLPACK format

  int loop_block_num_row = blockIdx.x * blockDim.x + threadIdx.x;
  int max_row =
      min((loop_block_num_row + 1) * num_per_block_row, Nrow);

  for (int loop_block_num_maxnzr = 0; loop_block_num_maxnzr < num_blocks_maxnzr;
       loop_block_num_maxnzr++) {
    int max_maxnzr =
        min((loop_block_num_maxnzr + 1) * num_per_block_maxnzr, maxnzr);
    for (int i = loop_block_num_row * num_per_block_row; i < max_row; i++) {
      double sum;
      if (loop_block_num_maxnzr == 0) {
        sum = 0;
      } else {
        sum = y[i];
      }
      for (int j = loop_block_num_maxnzr * num_per_block_maxnzr; j < max_maxnzr;
           j++) {
        int idx = i * maxnzr + j;
        sum += AS[idx] * x[JA[idx]];
      }
      y[i] = sum;
    }
  }
}

void ELLPACKMethodGPU::run() const {
  int num_threads = 32;
  int num_per_block_row = 8;
  int num_per_block_maxnzr = 100;

  int num_blocks_maxnzr =
      (maxnzr + num_per_block_maxnzr - 1) / num_per_block_maxnzr;

  int num_blocks_row =
    (Nrow + num_per_block_row - 1) / num_per_block_row;

    SpMv_gpu_thread_ELLPACK<<<(num_blocks_row + num_threads - 1) / num_threads,
    num_threads>>>(
        num_per_block_row, Nrow, num_per_block_maxnzr, num_blocks_maxnzr, maxnzr,
        AS, JA, x, y);
  cudaErrchk(cudaDeviceSynchronize());
  cudaErrchk(cudaPeekAtLastError());
}

void time_function(int iterations,
                   const SpMvMethod &method,
                   double *times, bool is_cuda) {
  for (int i = 0; i < iterations; ++i) {
    auto t1 = h_clock::now();
    method.run();
    if (is_cuda) {
      cudaErrchk(cudaDeviceSynchronize());
    }
    auto t2 = h_clock::now();
    double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();
    times[i] = time;
  }
}

void print_result(double *times, int iterations, const char *name) {
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

template <class T> void allocate_vector(T *&A, int n, MemoryType memory_type) {
  switch (memory_type) {
  case MemoryType::Host:
    A = new T[n];
    break;
  case MemoryType::Device:
    cudaErrchk(cudaMalloc(&A, n * sizeof(T)));
    break;
  case MemoryType::Unified:
    cudaErrchk(cudaMallocManaged(&A, n * sizeof(T)));
    break;
  }
}

template <class T>
void allocate_matrix(T **&A, int n, int m, MemoryType memory_type) {
  A = new T *[n];
  switch (memory_type) {
  case MemoryType::Host:
    A[0] = new T[n * m];
    break;
  case MemoryType::Device:
    cudaErrchk(cudaMalloc(&A[0], n * m * sizeof(T)));
    break;
  case MemoryType::Unified:
    cudaErrchk(cudaMallocManaged(&A[0], n * m * sizeof(T)));
    break;
  }
  for (int i = 0; i < n; ++i) {
    A[i] = A[0] + i * m;
  }
}
