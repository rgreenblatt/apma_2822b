#include <iostream>
#include <chrono>
#include <omp.h>
#include <sched.h>

using h_clock = std::chrono::high_resolution_clock;

int main() {
    int n_vals[2] = {512*1024*1024/8,  1024};
    int m_vals[2] = {10,  100000};
    for(int run = 0; run < 2; run++) {
        int n = n_vals[run];
        int m = m_vals[run];

        int num_threads = omp_get_max_threads();
        int chunk = (n + num_threads - 1) / num_threads;
        int offset[num_threads+1];

        for(int i = 0; i < num_threads; i++){
            offset[i] = chunk * i;
        }

        offset[num_threads] = n;

        double *a = new double[n];
        double *x = new double[n];
        double *y = new double[n];
        double *z = new double[n];
        double *w = new double[n];

        //Initialize values:
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int start = offset[tid];
            int end = offset[tid+1];
            for(int i = start; i < end; i++) {
                y[i] = 0.1 * i;
                z[i] = 0.2 * i;
                w[i] = 0.3 * i;
            }
        }

        auto t1_a = h_clock::now();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int start = offset[tid];
            int end = offset[tid+1];
            for (int iter = 0; iter < m; iter++){
                for(int i = start; i < end; i++) {
                    x[i] = y[i] + z[i] + y[i] + z[i];
                }
            }
        }
        auto t2_a = h_clock::now();

        auto time_span_a = std::chrono::duration_cast<std::chrono::duration<double>>(t2_a - t1_a);

        auto t1_b = h_clock::now();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int start = offset[tid];
            int end = offset[tid+1];
            for (int iter = 0; iter < m; iter++){
                for(int i = start; i < end; i++) {
                    x[i] = y[i] + z[i] - w[i];
                    a[i] = y[i] - z[i] + w[i];
                }
            }
        }
        auto t2_b = h_clock::now();

        auto time_span_b = std::chrono::duration_cast<std::chrono::duration<double>>(t2_b - t1_b);

        std::cout << "======  N: " << n << " ======" << std::endl;
        std::cout << "For a, total time per N: " << time_span_a.count() / m << " gigabyte per s: "
            << (n * m * sizeof(double) * 2) / (time_span_a.count() * (1024 * 1024 * 1024)) << std::endl;
        std::cout << "For b, total time per N: " << time_span_b.count() / m << " gigabyte per s: "
            << (n * m * sizeof(double) * 3) / (time_span_b.count() * (1024 * 1024 * 1024)) << std::endl;

        delete[] a;
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] w;
    }

    return 0;
}
