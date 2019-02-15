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

    int n_x = 20004;
    int n_y = 20004;

    double **A = new double*[n_x];
    double **B = new double*[n_x - 4];

    A[0] = new double[n_x * n_y];
    B[0] = new double[(n_x - 4) * (n_y - 4)];

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

        #pragma omp for
        for(int i = 0; i < n_x - 4; i++) {
            B[i] = B[0] + i * (n_y - 4);
        }
        
        #pragma omp for
        for(int i = 0; i < n_x; i++) {
            for(int j = 0; j < n_y; j++) {
                A[i][j] = f(i, j);
            }
        }
    }

    auto t1 = h_clock::now();

    int total = 0;
    #pragma omp parallel
    {
        int total_per = 0;
        #pragma omp for
        for(int i = 2; i < n_x - 2; i++) {
            for(int j = 2; j < n_y - 2; j++) {
                B[i-2][j-2] = (cm2 * A[i-2][j] +
                               cm1 * A[i-1][j] +
                               c_0 * A[i][j] +
                               cp1 * A[i+1][j] +
                               cp2 * A[i+2][j] +
                               dm2 * A[i][j-2] +
                               dm1 * A[i][j-1] +
                               dp1 * A[i][j+1] +
                               dp2 * A[i][j+2]) / 12;

            }
        }
    }
    

    auto t2 = h_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    int iters = (n_x - 4) * (n_y - 4);
    double total_bytes = (n_x * n_y + iters) * sizeof(double);

    std::cout << "|| standard || time per stencil: " << time.count() / iters * 1e6 << " micro sec  | GiB/s: " << 
        total_bytes / time.count() / 1024 / 1024 / 1024 << " | total: " << total << std::endl;

    int chunk_size_x = 500;
    int chunk_size_y = 100;
    int n_b_x = (n_x - 4) / chunk_size_x;
    int n_b_y = (n_y - 4) / chunk_size_y;

    t1 = h_clock::now();

    total = 0;
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for(int b_i = 0; b_i < n_b_x; b_i++) {
            for(int b_j = 0; b_j < n_b_y; b_j++) {
                for(int i = b_i * chunk_size_x; i < b_i * chunk_size_x + chunk_size_x; i++) {
                    for(int j = b_j * chunk_size_y; j < b_j * chunk_size_y + chunk_size_y; 
                            j++) {
                        int i_ = i + 2;
                        int j_ = j + 2;
                        B[i_-2][j_-2] = (cm2 * A[i_-2][j_] +
                                       cm1 * A[i_-1][j_] +
                                       c_0 * A[i_][j_] +
                                       cp1 * A[i_+1][j_] +
                                       cp2 * A[i_+2][j_] +
                                       dm2 * A[i_][j_-2] +
                                       dm1 * A[i_][j_-1] +
                                       dp1 * A[i_][j_+1] +
                                       dp2 * A[i_][j_+2]) / 12;
                    }
                }
            }
        }
    }
    

    t2 = h_clock::now();
    time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << "|| blocks   || time per stencil: " << time.count() / iters * 1e6 << " micro sec  | GiB/s: " << 
        total_bytes / time.count() / 1024 / 1024 / 1024 << " | total: " << total << std::endl;

    return 0;
}
