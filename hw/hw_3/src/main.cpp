#include <chrono>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <utility>
#include <vector>

// compile with g++ main.cpp -std=c++11 -fopenmp -O3 -march=native

using h_clock = std::chrono::high_resolution_clock;

void matrix_copy(double **from, double **to, int leading_dimension,
                 int other_dimension) {
  #pragma omp parallel for
  for (int i = 0; i < leading_dimension; i++) {
    for (int j = 0; j < other_dimension; j++) {
      to[i][j] = 2;
      to[i][j] = from[i][j];
    }
  }
}

void dgemm(double **A, double **B, int leading_dimension_a,
           int shared_dimension, int other_dimension_b, int num_threads,
           double **C) {

  int outer_block_size = leading_dimension_a / 16;
  int middle_block_size = shared_dimension / 8;
  int inner_block_size = other_dimension_b;
  int num_outer_per_thread = (int)ceil(((double)leading_dimension_a) /
                                       (outer_block_size * num_threads));
  int num_blocks_middle =
      (int)ceil(((double)shared_dimension) / (middle_block_size));
  int num_blocks_inner =
      (int)ceil(((double)other_dimension_b) / (inner_block_size));
  #pragma omp parallel
  {
    int thread = omp_get_thread_num();
    for (int b_outer = thread * num_outer_per_thread;
         b_outer < (thread + 1) * num_outer_per_thread; b_outer++) {
      int i_max =
          std::min(leading_dimension_a, (b_outer + 1) * outer_block_size);
      for (int b_middle = 0; b_middle < num_blocks_middle; b_middle++) {
        int j_max =
            std::min(shared_dimension, (b_middle + 1) * middle_block_size);
        for (int b_inner = 0; b_inner < num_blocks_inner; b_inner++) {
          int k_max =
              std::min(other_dimension_b, (b_inner + 1) * inner_block_size);
          for (int i = b_outer * outer_block_size; i < i_max; i++) {
            for (int j = b_middle * middle_block_size; j < j_max; j++) {
              for (int k = b_middle * middle_block_size; k < k_max; k++) {
                C[i][k] += A[i][j] * B[j][k];
              }
            }
          }
        }
      }
    }
  }
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int n_vals[3] = {2048, 4096, 8192};

  /**
   * assumptions:
   * - number of ranks can be square rooted
   * - n is divisible by the square root of the number of ranks
   */
  for (int run = 0; run < 3; run++) {
    // don't let any process start until the previous benchmark is finished
    MPI_Barrier(MPI_COMM_WORLD);

    int n = n_vals[run];

    int block_dim = (int)sqrt(world_size);
    int n_block = n / block_dim;

    double **A = new double *[n_block];
    double **B = new double *[n_block];
    double **C = new double *[n_block];

    double **working_A = new double *[n_block];  
    double **working_B = new double *[n_block];

    A[0] = new double[n_block * n_block];
    B[0] = new double[n_block * n_block];
    C[0] = new double[n_block * n_block];
    working_A[0] = new double[n_block * n_block];
    working_B[0] = new double[n_block * n_block];

    int rank_column = world_rank / block_dim;
    int rank_row = world_rank % block_dim;

    // Initialize values:
    int num_threads;
    #pragma omp parallel
    {
      if(omp_get_thread_num() == 0) {
          num_threads = omp_get_num_threads();
      }
      #pragma omp for
      for (int i = 0; i < n; i++) {
        A[i] = A[0] + i * n_block;
        B[i] = B[0] + i * n_block;
        C[i] = C[0] + i * n_block;
        working_A[i] = working_A[0] + i * n_block;
        working_B[i] = working_B[0] + i * n_block;
      }

      #pragma omp for
      for (int i = 0; i < n_block; i++) {
        for (int j = 0; j < n_block; j++) {
          int ii = i + n_block * rank_column;
          int jj = j + n_block * rank_row;
          A[i][j] = ii * 0.3 + jj * 0.4;
          B[i][j] = ii * 0.5 - jj * 0.3;
          C[i][j] = 0;
        }
      }
    }

    auto t1 = h_clock::now();

    MPI_Comm row_comm, column_comm;

    if (block_dim > 1) {
      MPI_Comm_split(MPI_COMM_WORLD, rank_row, rank_column, &row_comm);
      MPI_Comm_split(MPI_COMM_WORLD, rank_column, rank_row, &column_comm);
    }

    for (int i = 0; i < block_dim; i++) {
      // async send/receive: send to the next rank and recieve from the
      // previous rank in each column/row
      MPI_Request send_req_A, send_req_B, rec_req_A, rec_req_B;
      if (block_dim > 1) {
        MPI_Isend(A, n_block * n_block, MPI_DOUBLE,
                  ((rank_column + 1) % block_dim) * block_dim + rank_row, 0,
                  column_comm, &send_req_A);
        MPI_Isend(B, n_block * n_block, MPI_DOUBLE,
                  ((rank_row + 1) % block_dim) + block_dim * rank_column, 0,
                  row_comm, &send_req_B);
        MPI_Irecv(working_A, n_block * n_block, MPI_DOUBLE,
                  ((rank_column - 1 + block_dim) % block_dim) * block_dim +
                      rank_row,
                  0, column_comm, &rec_req_A);
        MPI_Irecv(working_B, n_block * n_block, MPI_DOUBLE,
                  ((rank_row - 1 + block_dim) % block_dim) +
                      block_dim * rank_column,
                  0, row_comm, &rec_req_B);
      }

      // perform matrix matrix multiplication on the process data
      dgemm(A, B, n_block, n_block, n_block, num_threads, C);

      if (block_dim > 1) {
        // wait for async send/rec so A/B can be copied into
        MPI_Status send_status_A, send_status_B, rec_status_A, rec_status_B;
        MPI_Wait(&send_req_A, &send_status_A);
        MPI_Wait(&rec_req_A, &rec_status_A);
        matrix_copy(working_A, A, n_block, n_block);
        MPI_Wait(&send_req_B, &send_status_B);
        MPI_Wait(&rec_req_B, &rec_status_B);
        matrix_copy(working_B, B, n_block, n_block);
      }
    }
    auto t2 = h_clock::now();

    double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();

    if (world_rank == 0) {
      // print out results
      std::cout << "======  N: " << n << " ======" << std::endl;

      // arithmetic is done this way to avoid issues with int overflow
      std::cout << "total time: " << time << " gigabyte per s: "
                << (n / 1024.) * (n / 1024.) / 1024. * sizeof(double) *
                       (3 / time)
                << " gflops per s: "
                << (n / 1024.) * (n / 1024.) * (n / 1024.) * (2 / time)
                << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
  }
  return 0;
}
