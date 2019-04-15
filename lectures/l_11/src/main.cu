#include <stdio.h>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <omp.h>
#include <cfloat>
#include <iostream>
#include <utility>

#define USE_NVTX

#ifdef USE_NVTX
#include "nvToolsExt.h"
#endif

struct ValLocPair {
  int val;
  int loc;
};

__host__ __device__ void get_nth(int N2, int *DATA, int nth_max, int offset,
                                 int *out) {
  if (nth_max == 0) {
    out[0] = DATA[0];
    out[1] = offset;
    return;
  } else if (nth_max == N2 - 1) {
    out[0] = DATA[N2 - 1];
    out[1] = N2 - 1 + offset;
    return;
  }
  int * lower_equal = new int[N2];
  int * greater = new int[N2];
  int size_lower_equal = 1;
  int size_greater = 0;
  greater[size_greater - 1] = DATA[0];

  for (int j = 1; j < N2; j++) {
    if (DATA[j] < DATA[0]) {
      size_lower_equal++;
      lower_equal[size_lower_equal - 1] = DATA[j];
    } else {
      size_greater++;
      greater[size_greater - 1] = DATA[j];
    }
  }
  if (size_lower_equal >= nth_max) {
    get_nth(size_lower_equal, lower_equal, nth_max, offset, out);
  } else {
    get_nth(size_greater, greater, nth_max, size_lower_equal, out);
  }
}

__global__ void find_MAX1(int N1, int N2, int **DATA, int *MAXES) {
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gtid == 0)
    printf("in find_MAX1\n");
}

__global__
void find_MAX2(int N1, int N2, int **DATA, int nmax,  int *MAXES){

   int gtid = threadIdx.x + blockIdx.x*blockDim.x;
   if (gtid==0) printf( "in find_MAX2\n" );
}

int main() {

  int ngpus = 0;
  cudaGetDeviceCount(&ngpus);
  printf("ngpus = %d\n", ngpus);
  if (ngpus > 0)
    cudaSetDevice(0);
  else
    return 0;

  int N1 = 10;
  int N2 = 100;

  std::vector<int> myvector;
  for (int i = 0; i < N2; ++i)
    myvector.push_back(i);

  int **DATA;
  int *MAXES;

#ifdef USE_NVTX
  // nvtxRangePushA("A");
  nvtxRangeId_t nvtx_1 = nvtxRangeStartA("A");
#endif

  cudaMallocManaged(&DATA, N1 * sizeof(int *));
  cudaMallocManaged(&DATA[0], N1 * N2 * sizeof(int));
  for (unsigned i = 1; i < N1; ++i)
    DATA[i] = DATA[0] + i * N2;

  cudaMallocManaged(&MAXES, N1 * 4 * sizeof(int));

#ifdef USE_NVTX
    nvtxRangeEnd(nvtx_1);
    //nvtxRangePop();
    #endif

    for (unsigned i = 0; i < N1; ++i){
       std::random_shuffle ( myvector.begin(), myvector.end() );
       for (unsigned j = 0; j < N2; ++j)
          DATA[i][j] = myvector[j];
    }

    int nth_max = std::rand() % N2;

/* ---------------  TASK 1  ------------ */

    for (int i = 0; i < N1; i++) {
      for (int j = 0; j < 2; j++) {
        MAXES[i * N2 + j * 2] = -INT_MAX;
        MAXES[i * N2 + j * 2 + 1] = -1;
      }
    }

#pragma omp parallel for
    for (int i = 0; i < N1; i++) {
      for (int j = 0; j < N1; j++) {
        if (DATA[i][j] > MAXES[i * N2]) {
          MAXES[i * N2] = DATA[i][j];
          MAXES[i * N2 + 1] = j;
        }
      }

    }


    #pragma omp parallel for
    for (int i = 0; i < N1; i++) {
      get_nth(N2, DATA[i], nth_max, 0, MAXES + i * N2 + 2);
    }


    std::cout << "==== cpu ====" << std::endl;
    for (int i = 0; i < N1; i++) {
      int base = i * N2;
      std::cout << "value: " << MAXES[base] << " loc: " << MAXES[base + 1]
                << " value nth: " << MAXES[i + base + 2]
                << " loc nth: " << MAXES[i + base + 3] << std::endl;
    }
    std::cout << std::endl;

    /* ---------------  TASK 2  ------------ */
/* ---------------  TASK 2  ------------ */

    //write GPU code to find the maximum in each row of DATA, i.e  MAX(DATA[i]) for each i
    //also find the locaiton of each maximum


    find_MAX1<<<1,1>>>(N1,N2, DATA, MAXES);


    //print results;

/* ---------------  TASK 3  ------------ */


    //write GPU code to find the first maximum and the Nth maximum value in each row of DATA, i.e  MAX(DATA[i]) for each i
    //also find the locaiton of each maximum

    find_MAX2<<<1,1>>>(N1,N2, DATA, nth_max, MAXES);

    //print results;
    cudaDeviceSynchronize();

    cudaFree(DATA[0]);
    cudaFree(DATA);

   return 0;

}
