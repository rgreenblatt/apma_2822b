#include <iostream>
#include <chrono>
#include <omp.h>
#include <Eigen/Dense>
#include <vector>

//compile with g++ main.cpp -std=c++11 -fopenmp -O3

using h_clock = std::chrono::high_resolution_clock;

int main() {
    long int n_vals[3] = {512, 1024, 2048};
    long int m = 1;
    for(int run = 0; run < 3; run++) {
        int n = n_vals[run];

        double **A = new double*[n];
        double **B = new double*[n];
        double **C = new double*[n];


        A[0] = new double[n * n];
        B[0] = new double[n * n];
        C[0] = new double[n * n];

        //Initialize values:
        #pragma omp parallel
        {
            #pragma omp for
            for(int i = 0; i < n; i++) {
                A[i] = A[0] + i * n;
                B[i] = B[0] + i * n;
                C[i] = C[0] + i * n;
            }

            #pragma omp for
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    A[i][j] = i * 0.3 + j * 0.4;
                    B[i][j] = i * 0.5 - j * 0.3;
                }
            }
        }

        //naive
        auto t1_n = h_clock::now();
        #pragma omp parallel
        {
            for (int iter = 0; iter < m; iter++) {
                #pragma omp for
                for(int i = 0; i < n; i++) {
                    for(int j = 0; j < n; j++) {
                        double sum = 0;
                        for(int k = 0; k < n; k++) {
                            sum += A[i][k] * C[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
        auto t2_n = h_clock::now();

        double time_n = std::chrono::duration_cast<std::chrono::duration<double>>(t2_n - t1_n).count();

        //collapse 2
        auto t1_c_2 = h_clock::now();
        #pragma omp parallel
        {
            for (int iter = 0; iter < m; iter++) {
                #pragma omp for
                for(int i = 0; i < n; i++) {
                    for(int j = 0; j < n; j++) {
                        double sum = 0;
                        for(int k = 0; k < n; k++) {
                            sum += A[i][k] * C[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
        auto t2_c_2 = h_clock::now();

        double time_c_2 = std::chrono::duration_cast<std::chrono::duration<double>>(t2_c_2 - t1_c_2).count();

        //eigen3 baseline
        Eigen::MatrixXd mat_A = Eigen::MatrixXd::Constant(n, n, 0.3);
        Eigen::MatrixXd mat_B = Eigen::MatrixXd::Constant(n, n, 0.3);
        Eigen::MatrixXd mat_C = Eigen::MatrixXd::Constant(n, n, 0.3);

        auto t1_e = h_clock::now();
        #pragma omp parallel
        {
            for (int iter = 0; iter < m; iter++) {
                mat_C = mat_A * mat_B;
            }
        }
        auto t2_e = h_clock::now();

        double time_e = std::chrono::duration_cast<std::chrono::duration<double>>(t2_e - t1_e).count();

        std::vector<std::string> method_names {"Naive", "Collapse 2", "Eigen3"};
        std::vector<double> method_times {time_n, time_c_2, time_e};
            
        std::cout << "======  N: " << n << " ======" << std::endl;

        for(int i = 0; i < method_times.size(); i++) {
            //arithmetic is done this way to avoid issues with int overflow
            std::cout << method_names[i] << " implementation, total time per N: " << method_times[i] / m << 
                " gigabyte per s: " << (n / 1024.) * (n / 1024.) * (m / 1024.) * 
                sizeof(double) * (3 / method_times[i])  << " gflops per s: " << (n / 1024.) * (n / 1024.) * 
                (n / 1024.) * m * (2 / method_times[i]) << std::endl;
        }

        delete[] A;
        delete[] B;
        delete[] C;
    }

    return 0;
}
