#pragma once

#include <assert.h>
#include <cblas.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <sstream>

/**
 * used for computing grid dimensions
 */
void closest_factors(int product, int &out_a, int &out_b) {
  out_b = (int)floor(sqrt(product));
  while (product % out_b) {
    out_b--;
  }
  out_a = product / out_b;
}

void matrix_copy(double **from, double **to, int leading_dimension,
                 int other_dimension) {
  #pragma omp parallel for
  for (int i = 0; i < leading_dimension; i++) {
    for (int j = 0; j < other_dimension; j++) {
      to[i][j] = from[i][j];
    }
  }
}

void dense_matrix_multiply(double **A, double **B, double **C, int size_i,
                           int size_j, int size_k, int num_threads,
                           bool use_blas = false) {

  if (use_blas) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size_i, size_k,
                size_j, 1, A[0], size_j, B[0], size_k, 1, C[0], size_k);
  } else {
    int outer_block_size = (size_i + 32) / 32;
    int middle_block_size = (size_j + 16) / 16;
    int inner_block_size = (size_k + 8) / 8;
    int num_outer_per_thread =
        (int)ceil(((double)size_i) / (outer_block_size * num_threads));
    int num_blocks_middle = (int)ceil(((double)size_j) / (middle_block_size));
    int num_blocks_inner = (int)ceil(((double)size_k) / (inner_block_size));
    #pragma omp parallel
    {
      int thread = omp_get_thread_num();
      for (int b_outer = thread * num_outer_per_thread;
           b_outer < (thread + 1) * num_outer_per_thread; b_outer++) {
        int i_max = std::min(size_i, (b_outer + 1) * outer_block_size);
        for (int b_middle = 0; b_middle < num_blocks_middle; b_middle++) {
          int j_max = std::min(size_j, (b_middle + 1) * middle_block_size);
          for (int b_inner = 0; b_inner < num_blocks_inner; b_inner++) {
            int k_max = std::min(size_k, (b_inner + 1) * inner_block_size);
            for (int i = b_outer * outer_block_size; i < i_max; i++) {
              for (int j = b_middle * middle_block_size; j < j_max; j++) {
                for (int k = b_inner * inner_block_size; k < k_max; k++) {
                  C[i][k] += A[i][j] * B[j][k];
                }
              }
            }
          }
        }
      }
    }
  }
}

void allocate_matrix(double **&A, int n, int m) {
  A = new double *[n];
  A[0] = new double[n * m];
  for (int i = 0; i < n; ++i) {
    A[i] = A[0] + i * m;
  }
}

void distributed_matrix_multiply(int size_i, int size_j, int size_k,
                                 int world_rank, int world_size,
                                 int num_threads,
                                 std::function<double(int, int)> f_a,
                                 std::function<double(int, int)> f_b,
                                 bool use_blas = false,
                                 bool verify_results = false) {
  int block_dim_row, block_dim_col;
  closest_factors(world_size, block_dim_col, block_dim_row);

  //determine where the rows of B will be distributed between processes
  int num_blocks_before_B_j[block_dim_row + 1];
  int num_blocks_per_B_j = (int)std::floor(block_dim_col / block_dim_row);
  int extra_blocks_B_j = block_dim_col - num_blocks_per_B_j * block_dim_row;
  num_blocks_before_B_j[0] = 0;
  for (int i = 1; i < block_dim_row + 1; ++i) {
    num_blocks_before_B_j[i] = num_blocks_before_B_j[i - 1] +
                               num_blocks_per_B_j +
                               (i <= extra_blocks_B_j ? 1 : 0);
  }

  if (verify_results) {
    assert(world_size == block_dim_row * block_dim_col);
    assert(block_dim_row <= block_dim_col);
    assert(block_dim_row > 0);

    assert(num_blocks_before_B_j[block_dim_row] == block_dim_col);
  }


  //the assumption is that matrix size is divisible by block dimensions
  //this assumption can be relaxed using padding or additional logic
  assert(size_i % block_dim_row == 0);
  assert(size_j % block_dim_col == 0);
  assert(size_j % block_dim_row == 0);
  assert(size_k % block_dim_col == 0);


  int rank_row = world_rank / block_dim_col;
  int rank_col = world_rank % block_dim_col;

  //determine which rank each process sends data to
  int rank_send_A = ((rank_col + 1) % block_dim_col) + rank_row * block_dim_col;
  int rank_rec_A = ((rank_col - 1 + block_dim_col) % block_dim_col) +
                   rank_row * block_dim_col;
  int rank_send_B = ((rank_row + 1) % block_dim_row) * block_dim_col + rank_col;
  int rank_rec_B =
      ((rank_row - 1 + block_dim_row) % block_dim_row) * block_dim_col +
      rank_col;

  int n_block_i = size_i / block_dim_row;
  int n_block_j = size_j / block_dim_col;
  int n_block_k = size_k / block_dim_col;

  double **A, **C, **working_A, **working_B;
  allocate_matrix(A, n_block_i, n_block_j);
  allocate_matrix(C, n_block_i, n_block_k);
  allocate_matrix(working_A, n_block_i, n_block_j);
  allocate_matrix(working_B, n_block_j, n_block_k);

  int num_blocks_B_j =
      num_blocks_before_B_j[rank_row + 1] - num_blocks_before_B_j[rank_row];

  double **all_B[num_blocks_B_j];
  for (int i = 0; i < num_blocks_B_j; ++i) {
    allocate_matrix(all_B[i], n_block_j, n_block_k);
  }

  // Initialize values:
  for (int i = 0; i < n_block_i; i++) {
    for (int j = 0; j < n_block_j; j++) {
      int ii = i + n_block_i * rank_row;
      int jj = (j + (num_blocks_before_B_j[rank_row] + rank_col) * n_block_j) %
               size_j;
      A[i][j] = f_a(ii, jj);
    }
    for (int k = 0; k < n_block_k; k++) {
      C[i][k] = 0;
    }
  }
  for (int which_block = 0; which_block < num_blocks_B_j; ++which_block) {
    for (int j = 0; j < n_block_j; j++) {
      int jj =
          j + n_block_j *
                  ((which_block + num_blocks_before_B_j[rank_row] + rank_col) %
                   block_dim_col);
      for (int k = 0; k < n_block_k; k++) {
        int kk = k + n_block_k * rank_col;
        all_B[which_block][j][k] = f_b(jj, kk);
      }
    }
  }

  for (int j = 0; j < block_dim_col; j++) {
    // async send/receive: send to the next rank and recieve from the
    // previous rank in each column/row
    MPI_Request send_req_A, send_req_B, rec_req_A, rec_req_B;

    if (block_dim_col > 1 && j < block_dim_col - 1) {
      MPI_Isend(A[0], n_block_i * n_block_j, MPI_DOUBLE, rank_send_A, 0,
                MPI_COMM_WORLD, &send_req_A);
      MPI_Irecv(working_A[0], n_block_i * n_block_j, MPI_DOUBLE, rank_rec_A, 0,
                MPI_COMM_WORLD, &rec_req_A);

      MPI_Isend(all_B[num_blocks_B_j - 1][0], n_block_j * n_block_k, MPI_DOUBLE,
                rank_send_B, 1, MPI_COMM_WORLD, &send_req_B);
      MPI_Irecv(working_B[0], n_block_j * n_block_k, MPI_DOUBLE, rank_rec_B, 1,
                MPI_COMM_WORLD, &rec_req_B);
    }

    // perform matrix matrix multiplication on the process data
    dense_matrix_multiply(A, all_B[0], C, n_block_i, n_block_j, n_block_k,
                          num_threads, use_blas);

    if (block_dim_col > 1 && j < block_dim_col - 1) {
      // wait for async send/rec so A/B can be copied into
      MPI_Status send_status_A, send_status_B, rec_status_A, rec_status_B;
      MPI_Wait(&send_req_A, &send_status_A);
      MPI_Wait(&rec_req_A, &rec_status_A);
      matrix_copy(working_A, A, n_block_i, n_block_j);
      MPI_Wait(&send_req_B, &send_status_B);
      for (int which_block = num_blocks_B_j - 1; which_block > 0;
           --which_block) {
        matrix_copy(all_B[which_block - 1], all_B[which_block], n_block_j,
                    n_block_k);
      }
      MPI_Wait(&rec_req_B, &rec_status_B);
      matrix_copy(working_B, all_B[0], n_block_j, n_block_k);
    }
  }

  delete[] A;
  for (int which_block = 1; which_block < num_blocks_B_j; ++which_block) {
    delete[] all_B[which_block];
  }
  if (verify_results) {

    MPI_Barrier(MPI_COMM_WORLD);

    double **A_test, **B_test, **C_test;
    allocate_matrix(A_test, size_i, size_j);
    allocate_matrix(B_test, size_j, size_k);
    allocate_matrix(C_test, size_i, size_k);
    #pragma omp parallel
    {
      #pragma omp for
      for (int i = 0; i < size_i; ++i) {
        for (int j = 0; j < size_j; ++j) {
          A_test[i][j] = f_a(i, j);
        }
        for (int k = 0; k < size_k; ++k) {
          C_test[i][k] = 0.;
        }
      }
      #pragma omp for
      for (int j = 0; j < size_j; ++j) {
        for (int k = 0; k < size_k; ++k) {
          B_test[j][k] = f_b(j, k);
        }
      }
    }
    dense_matrix_multiply(A_test, B_test, C_test, size_i, size_j, size_k,
                          num_threads, true);

    for (int i = 0; i < n_block_i; ++i) {
      int ii = i + rank_row * n_block_i;
      for (int k = 0; k < n_block_k; ++k) {
        int kk = k + rank_col * n_block_k;

        assert(std::abs(C[i][k] - C_test[ii][kk]) < 1e-6);
      }
    }
    std::cout << "verified" << std::endl;
  }
  delete[] C;
}
