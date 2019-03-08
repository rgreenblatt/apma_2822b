#include <iostream>
#include <chrono>
#include <omp.h>
#ifdef EIGEN
#include <Eigen/Dense>
#endif
#include <vector>
#include <math.h>

//compile with g++ main.cpp -std=c++11 -fopenmp -O3 -march=native

using h_clock = std::chrono::high_resolution_clock;

int main() {
    int n_vals[3] = {512, 1024, 2048};
    //int inner_block_sizes[3] = {256, 512, 256};
    int m = 3;
    for(int run = 0; run < 3; run++) {
        int n = n_vals[run];

        double **A = new double*[n];
        double **B = new double*[n];
        double **C = new double*[n];


        #ifdef EIGEN
        Eigen::MatrixXd mat_A(n, n);
        Eigen::MatrixXd mat_B(n, n);
        Eigen::MatrixXd mat_C(n, n);
        #endif

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
                    C[i][j] = 0;

                    #ifdef EIGEN
                    mat_A(i, j) = i * 0.3 + j * 0.4;
                    mat_B(i, j) = i * 0.5 - j * 0.3;
                    mat_C(i, j) = 0;
                    #endif
                }
            }
        }

        //naive
        auto t1_n = h_clock::now();
        for (int iter = 0; iter < m; iter++) {
            #pragma omp parallel for
            for(int i = 0; i < n; i++) {
                int flop_num_per_thread = 0;
                for(int j = 0; j < n; j++) {
                    for(int k = 0; k < n; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }
        auto t2_n = h_clock::now();

        double time_n = std::chrono::duration_cast<std::chrono::duration<double>>(t2_n - t1_n).count();
        
        //reset to all zeros
        #pragma omp parallel
        {
            #pragma omp for
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    C[i][j] = 0;
                }
            }
        }

        //blocking
        auto t1_c_2 = h_clock::now();

        int num_threads;
        #pragma omp parallel
        {
            if(omp_get_thread_num() == 0) {
                num_threads = omp_get_num_threads();
            }
        }

        int outer_block_size = 16;
        int middle_block_size = 32;
        int inner_block_size = 256;
        int num_outer_per_thread = ceil(((double) n) / (outer_block_size * num_threads));
        int num_blocks_middle = ceil(((double) n) / (middle_block_size));
        int num_blocks_inner = ceil(((double) n) / (inner_block_size));
        #pragma omp parallel
        {
            for (int iter = 0; iter < m; iter++) {
                int thread = omp_get_thread_num();
                for(int b_outer = thread * num_outer_per_thread; b_outer < (thread + 1) * 
                            num_outer_per_thread; b_outer++) {
                    for(int b_middle = 0;  b_middle < num_blocks_middle; b_middle++) {
                        for(int b_inner = 0;  b_inner < num_blocks_inner; b_inner++) {
                            for(int i = 0; i < outer_block_size; i++) {
                                for(int j = 0; j < middle_block_size; j++) {
                                    for(int k = 0; k < inner_block_size; k++) {
                                        int i_ = i + b_outer * outer_block_size;
                                        int j_ = j + b_middle * middle_block_size;
                                        int k_ = k + b_inner * inner_block_size;
                                        C[i_][k_] += A[i_][j_] * B[j_][k_];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        auto t2_c_2 = h_clock::now();

        double time_c_2 = std::chrono::duration_cast<std::chrono::duration<double>>(t2_c_2 - t1_c_2).count();

        std::vector<std::string> method_names {"Naive", "Optimized"};
        std::vector<double> method_times {time_n, time_c_2};
                    
        #ifdef EIGEN
        //eigen3 baseline
        auto t1_e = h_clock::now();
        for (int iter = 0; iter < m; iter++) {
            mat_C = mat_A * mat_B;
        }
        auto t2_e = h_clock::now();

        //check that the computation is valid using eigen
        //only a few values are checked, but that should be sufficient
        for(int i = n-4; i < n; i++) {
            for(int j = n-4; j < n; j++) {
                assert(abs(C[i][j] - mat_C(i, j)) < 1e0);
            }
        }

        double time_e = std::chrono::duration_cast<std::chrono::duration<double>>(t2_e - t1_e).count();

        method_names.push_back("Eigen3");
        method_times.push_back(time_e);
        #endif



            
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
