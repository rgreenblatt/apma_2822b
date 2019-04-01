#include <chrono>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <utility>
#include <vector>
#include "matrix.h"
#include "mkl.h"

// compile with g++ main.cpp -std=c++11 -fopenmp -O3 -march=native

using h_clock = std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int omp_num_threads = omp_get_max_threads();
  int mkl_num_threads = mkl_get_max_threads();
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0) {
    std::cout << "num threads found by omp: " << omp_num_threads
              << " num threads found by mkl: " << mkl_num_threads
              << " mpi nodes: " << world_size << std::endl << std::endl;
  }

  auto f_a = [](int i, int j) { return i * 0.3 + j * 0.4; };
  auto f_b = [](int j, int k) { return j * 0.5 + k * 0.6; };

  int size_i_vals[3] = {2048, 4096, 8192};
  int size_j_vals[3] = {2048, 4096, 8192};
  int size_k_vals[3] = {2048, 4096, 8192};

  for (int run = 0; run < 3; run++) {
    int size_i = size_i_vals[run];
    int size_j = size_j_vals[run];
    int size_k = size_k_vals[run];

    std::vector<std::pair<std::string, double>> algorithm_results;

    //testing with mkl dgemm
    auto t1 = h_clock::now();
    distributed_matrix_multiply(size_i, size_j, size_k, world_rank,
                                world_size, omp_num_threads, f_a, f_b, true);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t2 = h_clock::now();

    double time_mkl =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();
    algorithm_results.push_back(
        std::make_pair("mkl local, mine distributed", time_mkl));

    //testing with my matrix multiply
    t1 = h_clock::now();
    distributed_matrix_multiply(size_i, size_j, size_k, world_rank,
                                world_size, omp_num_threads, f_a, f_b, false);
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = h_clock::now();

    double time_mine =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();
    algorithm_results.push_back(std::make_pair("mine local, mine distributed", 
          time_mine));

    if (world_rank == 0) {
      // print out results
      std::cout << "======  size i: " << size_i << " size j: " << size_j
                << " size k: " << size_k << " ======" << std::endl;

      for(size_t i = 0; i < algorithm_results.size(); i++) {
        std::string name = algorithm_results[i].first;
        double time = algorithm_results[i].second;
        // arithmetic is done this way to avoid issues with int overflow
        double g_flops_per_s =
            (size_i / 1024.) * (size_j / 1024.) * (size_k / 1024.) * (2 / time);

        std::cout << "algorithm: " << name << " total time: " << time
                  << " gflops per s: " << g_flops_per_s << std::endl;
      }
      std::cout << std::endl;
    }
  }
  return 0;
}
