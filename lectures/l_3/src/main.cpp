#include <iostream>
#include <chrono>
#include <omp.h>
#include <sched.h>
#include <sstream>

using h_clock = std::chrono::high_resolution_clock;

double f(double x, double y) {
    return x * x + y * y;
}

int main() {
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        std::stringstream s;
        s << "num threads: " << omp_get_num_threads() << " thread id: " << omp_get_thread_num() << std::endl;
        std::cout << s.str();
    }

    int n_x = 20000;
    int n_y = 20000;

    double **U = new double*[n_x];
    double **d2U = new double*[n_x-1];
    double *x = new double[n_x];
    double *y = new double[n_y];

    U[0] = new double[n_x * n_y];
    d2U[0] = new double[(n_x - 1) * (n_y - 1)];

    auto t1 = h_clock::now();

    #pragma omp parallel
    {
        #pragma omp for nowait
        for(int i = 0; i < n_x; i++) {
            U[i] = U[0] + i * n_y;
            x[i] = i;
        }
        #pragma omp for nowait
        for(int i = 0; i < n_x - 1; i++) {
            d2U[i] = d2U[0] + i * n_y;
        }

        #pragma omp for
        for(int i = 0; i < n_y; i++) {
            y[i] = i;
        }

        #pragma omp for
        for(int i = 1; i < n_x - 1; i++) {
            for(int j = 1; j < n_y - 1; j++) {
                U[i][j] = f(x[i], y[j]);
            }
        }

        #pragma omp for
        for(int i = 1; i < n_x - 1; i++) {
            for(int j = 1; j < n_y - 1; j++) {
                d2U[i][j] = (U[i][j-1] - 2 * U[i][j] + U[i][j+1])  + 
                            (U[i-1][j] - 2 * U[i][j] + U[i+1][j]);
            }
        }
    }

    auto t2 = h_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << "time: " << time.count() << std::endl;

    return 0;
}
