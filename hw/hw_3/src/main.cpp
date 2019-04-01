#include <chrono>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <utility>
#include <vector>
#include <assert.h>
#include <cmath>
#include "mkl.h"

extern "C" void cblas_dgemm (const CBLAS_LAYOUT Layout, 
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, 
    const MKL_INT m, const MKL_INT n, const MKL_INT k, const double alpha, 
    const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb, 
    const double beta, double *c, const MKL_INT ldc);


// compile with g++ main.cpp -std=c++11 -fopenmp -O3 -march=native

using h_clock = std::chrono::high_resolution_clock;

void matrix_copy(double **from, double **to, int leading_dimension,
                 int other_dimension) {
  #pragma omp parallel for
  for (int i = 0; i < leading_dimension; i++) {
    for (int j = 0; j < other_dimension; j++) {
      to[i][j] = from[i][j];
    }
  }
}

void dgemm(double **A, double **B, int leading_dimension_a,
           int shared_dimension, int other_dimension_b, int num_threads,
           double **C, bool use_mkl=false) {

  if(use_mkl) {
    omp_set_num_threads(num_threads);
    mkl_set_num_threads(num_threads);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,  
        leading_dimension_a, shared_dimension, other_dimension_b, 1, 
        A[0], shared_dimension, B[0], other_dimension_b, 0, C[0], 
        other_dimension_b);
  } else {
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
      /* std::cout << "in thread: " << thread << std::endl; */
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
}

//source: 
//https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
int ipow(int base, int exp)
{
  int result = 1;
  for (;;)
  {
    if (exp & 1)
      result *= base;
    exp >>= 1;
    if (!exp)
      break;
    base *= base;
  }

  return result;
}

void closest_factors(int product, int& out_a, int& out_b) {
  out_b = (int) floor(sqrt(product));
  while(product % out_b) {
    out_b--;
  }
  out_a = product / out_b;
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int omp_num_threads = omp_get_max_threads();
  int mkl_num_threads = mkl_get_max_threads();

  int world_size, world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  if(world_rank == 0) {
    std::cout << "num threads found by omp: " << omp_num_threads << 
      " num threads found by mkl: " << mkl_num_threads << " mpi nodes: " 
      << world_size << std::endl;
  }

  int block_dim_row, block_dim_col;
  closest_factors(world_size, block_dim_col, block_dim_row);

  assert(world_size ==  block_dim_row * block_dim_col);
  assert(block_dim_row <= block_dim_col);
  //TODO
  int block_dim_j = std::max(block_dim_row, block_dim_col);

  int n_i_vals[3] = {2048, 4096, 8192};
  int n_j_vals[3] = {2048, 4096, 8192};
  int n_k_vals[3] = {2048, 4096, 8192};

  int num_blocks_before_B_j[block_dim_row + 1];
  assert(block_dim_row > 0);
  num_blocks_before_B_j[0] = 0;
  int num_blocks_per_B_j = (int) std::floor(block_dim_j / block_dim_row);
  int extra_blocks_B_j = num_blocks_per_B_j * block_dim_row - block_dim_j;

  for (int i = 1; i < block_dim_row + 1; ++i) {
    num_blocks_before_B_j[i] =
        num_blocks_before_B_j[i - 1] + num_blocks_per_B_j + (i < extra_blocks_B_j
            ? 1
            : 0);
  }

  assert(num_blocks_before_B_j[block_dim_row] == block_dim_j);

  int rank_row = world_rank / block_dim_col;
  int rank_col = world_rank % block_dim_col;

  int num_blocks_B_j =
      num_blocks_before_B_j[rank_row + 1] - num_blocks_before_B_j[rank_row];

  for (int run = 0; run < 3; run++) {
    int n_i = n_i_vals[run];
    int n_j = n_j_vals[run];
    int n_k = n_k_vals[run];

    //assumption is that matrix size is divisible by block dimensions
    //this assumption can be relaxed using padding or additional logic
    assert(n_i % block_dim_row == 0);
    assert(n_j % block_dim_col == 0);
    assert(n_j % block_dim_row == 0);
    assert(n_k % block_dim_col == 0);

    int n_block_i = n_i / block_dim_row;
    int n_block_j = n_j / block_dim_col;
    int n_block_k = n_k / block_dim_col;

    double **A = new double *[n_block_i];
    double** all_B[num_blocks_B_j];
    for (int i = 0; i < num_blocks_B_j; ++i) {
      all_B[i] = new double *[n_block_j];
    }
    double **C = new double *[n_block_i];

    double **working_A = new double *[n_block_i];  
    double **working_B = new double *[n_block_j];

    A[0] = new double[n_block_i * n_block_j];
    for (int i = 0; i < num_blocks_B_j; ++i) {
      all_B[i][0] = new double[n_block_j * n_block_k];
    }
    C[0] = new double[n_block_i * n_block_k];
    working_A[0] = new double[n_block_i * n_block_j];
    working_B[0] = new double[n_block_j * n_block_k];

    //Initialize values:
    //The effective index in terms of initial data is not the same as the rank
    //in the j dimension. This staggers the data to make all multiplications
    //valid. For B, blocks in the j dimension must be aligned with the columns,
    //so this just requires a single offset

    int num_threads;
    #pragma omp parallel
    {
      if(omp_get_thread_num() == 0) {
          num_threads = omp_get_num_threads();
      }
    }

    /* std::cout << "running with: " << num_threads << std::endl; */

    #pragma omp parallel
    {
      for (int which_block = 0; which_block < num_blocks_B_j; ++which_block) {
        for (int j = 0; j < n_block_j; ++j) {
          all_B[which_block][j] = all_B[which_block][0] + j * n_block_k;
        }
      }
      #pragma omp for
      for (int j = 0; j < n_block_j; ++j) {
        working_B[j] = working_B[0] + j * n_block_k;
      }

      for (int i = 0; i < n_block_i; i++) {
        A[i] = A[0] + i * n_block_j;
        working_A[i] = working_A[0] + i * n_block_j;
        C[i] = C[0] + i * n_block_k;
      }

      #pragma omp for
      for (int i = 0; i < n_block_i; i++) {
        for (int j = 0; j < n_block_j; j++) {
          int ii = i + n_block_i * rank_row;
          int jj = (j + num_blocks_before_B_j[rank_row] * n_block_j) % n_j;
          A[i][j] = ii * 0.3 + jj * 0.4;
        }
        for (int k = 0; k < n_block_k; k++) {
          C[i][k] = 0;
        }
      }
      for(int which_block = 0; which_block < num_blocks_B_j; ++which_block) {
        #pragma omp for
        for (int j = 0; j < n_block_j; j++) {
          int jj =
              j + n_block_j * (which_block + num_blocks_before_B_j[rank_row]);
          for (int k = 0; k < n_block_k; k++) {
            int kk = k + n_block_k * rank_col;
            all_B[which_block][j][k] = jj * 0.5 + kk * 0.6;
          }
        }
      }
    }

    auto t1 = h_clock::now();

    /* MPI_Comm row_comm, column_comm; */

    /* if (block_dim > 1) { */
    /*   MPI_Comm_split(MPI_COMM_WORLD, rank_row, world_rank, &row_comm); */
    /*   MPI_Comm_split(MPI_COMM_WORLD, rank_column, world_rank, &column_comm); */
    /* } */

    for (int j = 0; j < block_dim_j; j++) {
      // async send/receive: send to the next rank and recieve from the
      // previous rank in each column/row
      MPI_Request send_req_A, send_req_B, rec_req_A, rec_req_B;

      if (block_dim_j > 1) {
        /* std::cout << "multiple nodes, sending" << std::endl; */
        int rank_send_A = ((rank_col + 1) % block_dim_col) + rank_row * 
          block_dim_row;
        int rank_rec_A = ((rank_col - 1 + block_dim_col) % block_dim_col) + 
          rank_row * block_dim_row;
        int rank_send_B = ((rank_row + 1) % block_dim_row) * block_dim_row + 
          rank_col;
        int rank_rec_B = ((rank_row - 1 + block_dim_row) % block_dim_row) * 
          block_dim_row + rank_col;

        /* std::cout << */ 
        /*   "rank_col:" << rank_col << " block_dim_col: " << block_dim_col << */ 
        /*   "rank_row:" << rank_row << " block_dim_row: " << block_dim_row << */ 
        /*    std::endl; */

        /* std::cout << "rank:" << world_rank << " s A: " << rank_send_A << */ 
        /*   " s B: " << rank_send_B << " r A: " << rank_rec_A << " r B: " << */ 
        /*   rank_rec_B << std::endl; */

        //A can always be sent as entire blocks
        MPI_Isend(A[0], n_block_i * n_block_j, MPI_DOUBLE, rank_send_A, 0,
                  MPI_COMM_WORLD, &send_req_A);
        MPI_Irecv(working_A[0], n_block_i * n_block_j, MPI_DOUBLE, 
                  rank_rec_A, 0, MPI_COMM_WORLD, &rec_req_A);

        MPI_Isend(all_B[num_blocks_B_j - 1][0], n_block_j * n_block_k, MPI_DOUBLE,
                  rank_send_B, 1, MPI_COMM_WORLD, &send_req_B);
        MPI_Irecv(working_B[0], n_block_j * n_block_k, MPI_DOUBLE,
                  rank_rec_B, 1, MPI_COMM_WORLD, &rec_req_B);
      }

      /* std::cout << "after send: " << world_rank << std::endl; */

      // perform matrix matrix multiplication on the process data
      dgemm(A, all_B[0], n_block_i, n_block_j, n_block_k, num_threads, C, true);

      /* std::cout << "after dgemm: " << world_rank << std::endl; */
      if (block_dim_j > 1) {
        // wait for async send/rec so A/B can be copied into
        MPI_Status send_status_A, send_status_B, rec_status_A, rec_status_B;
        /* std::cout << "after wait -1:" << world_rank << std::endl; */
        MPI_Wait(&send_req_A, &send_status_A);
        /* std::cout << "after wait 0: " << world_rank << std::endl; */
        MPI_Wait(&rec_req_A, &rec_status_A);
        /* std::cout << "after wait 1: " << world_rank << std::endl; */
        matrix_copy(working_A, A, n_block_i, n_block_j);
        /* std::cout << "after copy: " << world_rank << std::endl; */
        MPI_Wait(&send_req_B, &send_status_B);
        for (int which_block = 1; which_block < num_blocks_B_j; ++which_block) {
          matrix_copy(all_B[which_block - 1], all_B[which_block], n_block_j,
                      n_block_k);
        }
        MPI_Wait(&rec_req_B, &rec_status_B);
        matrix_copy(working_B, all_B[0], n_block_j, n_block_k);
      }
      /* std::cout << "after loop" << std::endl; */
    }

    // don't measure the time until all processes finish
    MPI_Barrier(MPI_COMM_WORLD);
    auto t2 = h_clock::now();

    double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();

    if (world_rank == 0) {
      // print out results
      std::cout << "======  n_i: " << n_i << " n_j: " << n_j << " n_k: " << n_k
                << " ======" << std::endl;

      // arithmetic is done this way to avoid issues with int overflow
      std::cout << "total time: " << time << " gflops per s: "
                << (n_i / 1024.) * (n_j / 1024.) * (n_k / 1024.) * (2 / time)
                << std::endl;
    }

    delete[] A;
      for (int which_block = 1; which_block < num_blocks_B_j; ++which_block) {
        delete[] all_B[which_block];
      }
    delete[] C;
  }
  return 0;
}
