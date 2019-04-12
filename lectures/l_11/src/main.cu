#include <stdio.h>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <omp.h>
#include <cfloat>
#include <iostream>

#define USE_NVTX

#ifdef USE_NVTX
#include "nvToolsExt.h"
#endif



__global__
void find_MAX1(int N1, int N2, int **DATA, int **MAX_VALS, int **MAX_LOCS){
   int gtid = threadIdx.x + blockIdx.x*blockDim.x;
   if (gtid==0) printf( "in find_MAX1\n" );
}

__global__
void find_MAX2(int N1, int N2, int **DATA, int nmax,  int **MAX_VALS, int **MAX_LOCS){

   int gtid = threadIdx.x + blockIdx.x*blockDim.x;
   if (gtid==0) printf( "in find_MAX2\n" );
}



int main(){

    int ngpus = 0;
    cudaGetDeviceCount(&ngpus);
    printf("ngpus = %d\n",ngpus);
    if (ngpus > 0)    cudaSetDevice(0);
    else return 0;

    int N1 = 10;
    int N2 = 100;

    std::vector<int> myvector;
    for (int i=0; i<N2; ++i) myvector.push_back(i);

    int **DATA;
    int **MAX_VALS, **MAX_LOCS;

    #ifdef USE_NVTX
    //nvtxRangePushA("A");
    nvtxRangeId_t nvtx_1 = nvtxRangeStartA("A");
    #endif


    cudaMallocManaged(&DATA,N1*sizeof(int*));
    cudaMallocManaged(&DATA[0],N1*N2*sizeof(int));
    for (unsigned i = 1; i < N1; ++i)
      DATA[i] = DATA[0] + i*N2;

    #ifdef USE_NVTX
    nvtxRangeEnd(nvtx_1);
    //nvtxRangePop();
    #endif

    for (unsigned i = 0; i < N1; ++i){
       std::random_shuffle ( myvector.begin(), myvector.end() );
       for (unsigned j = 0; j < N2; ++j)
          DATA[i][j] = myvector[j];
    }

    int nmax = std::rand() % N2;

/* ---------------  TASK 1  ------------ */

    for (int i = 0; i < N1; i++) {
      for (int j = 0; j < 2; j++) {
        MAX_VALS[i][j] = -INT_MAX;
        MAX_LOCS[i][j] = -1;
      }
    }

#pragma omp parallel for
    for (int i = 0; i < N1; i++) {
      for (int j = 0; j < N1; j++) {
        if (DATA[i][j] > MAX_VALS[i]) {
          MAX_VALS[i] = DATA[i][j];
          MAX_LOCS[i] = j;
        }
      }
    }

    std::cout << "==== cpu ====" << std::endl;
    for (int i = 0; i < N1; i++) {
      std::cout << "value: " << MAX_VALS[i] << " loc: " << MAX_LOCS[i]
                << std::endl;
    }
    std::cout << std::endl;

/* ---------------  TASK 2  ------------ */

    //write GPU code to find the maximum in each row of DATA, i.e  MAX(DATA[i]) for each i
    //also find the locaiton of each maximum


    find_MAX1<<<1,1>>>(N1,N2, DATA, MAX_VALS, MAX_LOCS);


    //print results;

/* ---------------  TASK 3  ------------ */


    //write GPU code to find the first maximum and the Nth maximum value in each row of DATA, i.e  MAX(DATA[i]) for each i
    //also find the locaiton of each maximum

    find_MAX2<<<1,1>>>(N1,N2, DATA, nmax, MAX_VALS, MAX_LOCS);

    //print results;

    cudaFree(DATA[0]);
    cudaFree(DATA);

   return 0;

}
