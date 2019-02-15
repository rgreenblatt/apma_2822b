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

    double **A = new double*[n_x];
    double **B = new double*[n_x - 4];

    A[0] = new double[n_x * n_y];
    B[0] = new double[(n_x - 4) * (n_y - 4)];

    auto t1 = h_clock::now();

    double cm2 = -1.;
    double cm1 = 16.;
    double c_0 = -60.;
    double cp1 = 16.;
    double cp2 = -1.;
    double dm2 = -1.;
    double dm1 = 16.;
    double dp1 = 16.;
    double dp2 = -1.;

    #pragma omp parallel
    {
        #pragma omp for nowait
        for(int i = 0; i < n_x; i++) {
            A[i] = A[0] + i * n_y;
        }

        #pragma omp for nowait
        for(int i = 0; i < n_x - 4; i++) {
            B[i] = B[0] + i * (n_y - 4);
        }
        
        #pragma omp for
        for(int i = 0; i < n_x; i++) {
            for(int j = 0; j < n_y; j++) {
                A[i][j] = f(i, j);
            }
        }


        #pragma omp
        for(int i = 2; i < n_x - 2; i++) {
            for(int j = 2; j < n_y - 2; j++) {
                B[i-2][j-2] = cm2 * A[i-2][j] +
                          cm1 * A[i-1][j] +
                          c_0 * A[i][j] +
                          cp1 * A[i+1][j] +
                          cp2 * A[i+2][j] +
                          dm2 * A[i][j-2] +
                          dm1 * A[i][j-1] +
                          dp1 * A[i][j+1] +
                          dp2 * A[i][j+2];

            }
        }
    }

    auto t2 = h_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << "time: " << time.count() << std::endl;

    return 0;
}
