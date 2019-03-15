#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <stdio.h>

__global__
void triad_kernel(int N, double *x, double *y, double *z) {
    for(unsigned i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i+=blockDim.x*gridDim.x)
        z[i] = x[i] + y[i];
}

int main() {
    int N = 100;
    double *x, *y, *z;
    cudaMallocManaged((void**) &x, N*sizeof(double));
    cudaMallocManaged((void**) &y, N*sizeof(double));
    cudaMallocManaged((void**) &z, N*sizeof(double));

    for(unsigned i = 0; i < N; i++) {
        x[i] = 0.001*i; y[i] = 0.03*i;
    }

    int nthreads = 64;
    int nblocks = (N+nthreads-1)/ nthreads;
    triad_kernel<<<nblocks, nthreads>>>(N,x,y,z);
    cudaDeviceSynchronize();

    for(unsigned i = 0; i < N; i++) {
        printf("z[%d] = %g; expected z = %g\n|" ,i,z[i],x[i]+y[i]);
    }

    cudaFree(x); cudaFree(y); cudaFree(z);

    return 0;
}

