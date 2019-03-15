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
    printf("dkfjdkfj");
    int N = 100;
    double *x, *y, *z, *x_d, *y_d, *z_d;
    cudaMalloc((void**) &x_d, N*sizeof(double));
    cudaMalloc((void**) &y_d, N*sizeof(double));
    cudaMalloc((void**) &z_d, N*sizeof(double));

    for(unsigned i = 0; i < N; i++) {
        x[i] = 0.001*i; y[i] = 0.03*i;
    }

    cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);

    int nthreads = 64;
    int nblocks = (N+nthreads-1)/ nthreads;
    triad_kernel<<<nblocks, nthreads>>>(N,x_d,y_d,z_d);
    cudaDeviceSynchronize();

    cudaMemcpy(z, z_d, N*sizeof(double), cudaMemcpyDeviceToHost);
    for(unsigned i = 0; i < N; i++) {
        printf("z[%d] = %g; expected z = %g\n|" ,i,z[i],x[i]+y[i]);
    }

    cudaFree(x_d); cudaFree(y_d); cudaFree(z_d);
    delete[] x; delete[] y; delete[] z;

    return 0;
}

