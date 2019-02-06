#include <iostream>
#include <chrono>
#include <omp.h>
#include <sched.h>
#include <sstream>

#define PARALLEL

using h_clock = std::chrono::high_resolution_clock;

int main(){
    int n = 5000;
    int m = 100000;
    int num_threads = omp_get_max_threads();
    int chunk = (n + num_threads - 1) / num_threads;
    int offset[num_threads+1];

    for(int i = 0; i < num_threads; i++){
        offset[i] = chunk * i;
    }

    offset[num_threads] = n;

    double *x;
    x = new double[n];

    auto t1 = h_clock::now();
    #ifdef PARALLEL
    #pragma omp parallel
    #endif
    {
        int tid = omp_get_thread_num();
        int start = offset[tid];
        int end = offset[tid+1];
        for (int iter = 0; iter < m; iter++){
            for(int i = start; i < end; i++) {
                x[i] = 0.1 * i;
            }
        }
    }
    auto t2 = h_clock::now();

    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << "gigabyte per s: " << (n * m * sizeof(double)) / (time_span.count() * (1024 * 1024 * 1024)) << std::endl;

    delete[] x;

    return 0;
}
