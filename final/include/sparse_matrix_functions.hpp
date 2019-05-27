/*
//@HEADER
// ************************************************************************
//
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#pragma once

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>

#include "cuda_sparse_matrix_functions.hpp"
#include "elem_data.hpp"
#include "exchange_externals.hpp"
#include "matrix_copy_op.hpp"
#include "matrix_init_op.hpp"
#include "my_timer.hpp"
#include "vector.hpp"
#include "vector_functions.hpp"

#ifdef MINIFE_HAVE_TBB
#include "locking_matrix.hpp"
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace miniFE {

template <typename MatrixType>
void init_matrix(MatrixType &M, const std::vector<int> &rows,
                 const std::vector<int> &row_offsets,
                 const std::vector<int> &row_coords, int global_nodes_x,
                 int global_nodes_y, int global_nodes_z, int global_nrows,
                 const simple_mesh_description &mesh) {
  MatrixInitOp<MatrixType> mat_init(rows, row_offsets, row_coords,
                                    global_nodes_x, global_nodes_y,
                                    global_nodes_z, global_nrows, mesh, M);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(mat_init.n); ++i) {
    mat_init(i);
  }
}

template <typename T, typename U>
void sort_with_companions(ptrdiff_t len, T *array, U *companions) {
  ptrdiff_t i, j, index;
  U companion;

  for (i = 1; i < len; i++) {
    index = array[i];
    companion = companions[i];
    j = i;
    while ((j > 0) && (array[j - 1] > index)) {
      array[j] = array[j - 1];
      companions[j] = companions[j - 1];
      j = j - 1;
    }
    array[j] = static_cast<T>(index);
    companions[j] = companion;
  }
}

template <typename MatrixType>
void write_matrix(const std::string &filename, MatrixType &mat) {
  typedef typename MatrixType::ScalarType ScalarType;

  int numprocs = 1, myproc = 0;
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  std::ostringstream osstr;
  osstr << filename << "." << numprocs << "." << myproc;
  std::string full_name = osstr.str();
  std::ofstream ofs(full_name.c_str());

  size_t nrows = mat.rows.size();
  size_t nnz = mat.num_nonzeros();

  for (int p = 0; p < numprocs; ++p) {
    if (p == myproc) {
      if (p == 0) {
        ofs << nrows << " " << nnz << std::endl;
      }
      for (size_t i = 0; i < nrows; ++i) {
        size_t row_len = 0;
        int *cols = NULL;
        ScalarType *coefs = NULL;
        mat.get_row_pointers(mat.rows[i], row_len, cols, coefs);

        for (size_t j = 0; j < row_len; ++j) {
          ofs << mat.rows[i] << " " << cols[j] << " " << coefs[j] << std::endl;
        }
      }
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
}

template <typename Scalar>
void sum_into_row(size_t row_len, int *row_indices, Scalar *row_coefs,
                  size_t num_inputs, const int *input_indices,
                  const Scalar *input_coefs) {
  for (size_t i = 0; i < num_inputs; ++i) {
    int *loc =
        std::lower_bound(row_indices, row_indices + row_len, input_indices[i]);
    if (loc - row_indices < static_cast<int>(row_len) &&
        *loc == input_indices[i]) {
      row_coefs[loc - row_indices] += input_coefs[i];
    }
  }
}

template <typename MatrixType>
void sum_into_row(int row, size_t num_indices, const int *col_inds,
                  const typename MatrixType::ScalarType *coefs,
                  MatrixType &mat) {
  typedef typename MatrixType::ScalarType Scalar;

  size_t row_len = 0;
  int *mat_row_cols = NULL;
  Scalar *mat_row_coefs = NULL;

  mat.get_row_pointers(row, row_len, mat_row_cols, mat_row_coefs);
  if (row_len == 0)
    return;

  sum_into_row(row_len, mat_row_cols, mat_row_coefs, num_indices, col_inds,
               coefs);
}

template <typename MatrixType>
void sum_in_symm_elem_matrix(size_t num, const int *indices,
                             const typename MatrixType::ScalarType *coefs,
                             MatrixType &mat) {
  typedef typename MatrixType::ScalarType Scalar;

  // indices is length num (which should be nodes-per-elem)
  // coefs is the upper triangle of the element diffusion matrix
  // which should be length num*(num+1)/2

  size_t row_offset = 0;
  for (size_t i = 0; i < num; ++i) {
    int row = indices[i];

    const Scalar *row_coefs = &coefs[row_offset];
    const int *row_col_inds = &indices[i];
    size_t row_len = num - i;
    row_offset += row_len;

    size_t mat_row_len = 0;
    int *mat_row_cols = NULL;
    Scalar *mat_row_coefs = NULL;

    mat.get_row_pointers(row, mat_row_len, mat_row_cols, mat_row_coefs);
    if (mat_row_len == 0) {
      continue;
    }

    sum_into_row(mat_row_len, mat_row_cols, mat_row_coefs, row_len,
                 row_col_inds, row_coefs);

    size_t offset = i;
    for (size_t j = 0; j < i; ++j) {
      Scalar coef = coefs[offset];
      sum_into_row(mat_row_len, mat_row_cols, mat_row_coefs, 1, &indices[j],
                   &coef);
      offset += num - (j + 1);
    }
  }
}

template <typename MatrixType>
void sum_in_elem_matrix(size_t num, const int *indices,
                        const typename MatrixType::ScalarType *coefs,
                        MatrixType &mat) {
  size_t offset = 0;

  for (size_t i = 0; i < num; ++i) {
    sum_into_row(indices[i], num, &indices[0], &coefs[offset], mat);
    offset += num;
  }
}

template <typename Scalar, typename MatrixType, typename VectorType>
void sum_into_global_linear_system(ElemData<Scalar> &elem_data, MatrixType &A,
                                   VectorType &b) {

  sum_in_symm_elem_matrix(elem_data.nodes_per_elem, elem_data.elem_node_ids,
                          elem_data.elem_diffusion_matrix, A);

  sum_into_vector(elem_data.nodes_per_elem, elem_data.elem_node_ids,
                  elem_data.elem_source_vector, b);
}

#ifdef MINIFE_HAVE_TBB
template <typename MatrixType>
void sum_in_elem_matrix(size_t num, const int *indices,
                        const typename MatrixType::ScalarType *coefs,
                        LockingMatrix<MatrixType> &mat) {
  size_t offset = 0;

  for (size_t i = 0; i < num; ++i) {
    mat.sum_in(indices[i], num, &indices[0], &coefs[offset]);
    offset += num;
  }
}

template <typename Scalar, typename MatrixType,
          typename VectorType>
void sum_into_global_linear_system(ElemData<Scalar> &elem_data,
                                   LockingMatrix<MatrixType> &A,
                                   LockingVector<VectorType> &b) {
  sum_in_elem_matrix(elem_data.nodes_per_elem, elem_data.elem_node_ids,
                     elem_data.elem_diffusion_matrix, A);
  sum_into_vector(elem_data.nodes_per_elem, elem_data.elem_node_ids,
                  elem_data.elem_source_vector, b);
}
#endif

template <typename MatrixType>
void add_to_diagonal(typename MatrixType::ScalarType value, MatrixType &mat) {
  for (size_t i = 0; i < mat.rows.size(); ++i) {
    sum_into_row(mat.rows[i], 1, &mat.rows[i], &value, mat);
  }
}

template <typename MatrixType>
double parallel_memory_overhead_MB(const MatrixType &A) {
  double mem_MB = 0;

#ifdef HAVE_MPI
  double invMB = 1.0 / (1024 * 1024);
  mem_MB = invMB * static_cast<double>(A.external_index.size() * sizeof(int));
  mem_MB +=
      invMB * static_cast<double>(A.external_local_index.size() * sizeof(int));
  mem_MB +=
      invMB * static_cast<double>(A.elements_to_send.size() * sizeof(int));
  mem_MB += invMB * static_cast<double>(A.neighbors.size() * sizeof(int));
  mem_MB += invMB * static_cast<double>(A.recv_length.size() * sizeof(int));
  mem_MB += invMB * static_cast<double>(A.send_length.size() * sizeof(int));

  double tmp = mem_MB;
  MPI_Allreduce(&tmp, &mem_MB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  return mem_MB;
}

template <typename MatrixType>
void rearrange_matrix_local_external(MatrixType &A) {
  // This function will rearrange A so that local entries are contiguous at the
  // front of A's memory, and external entries are contiguous at the back of A's
  // memory.
  //
  // A.row_offsets will describe where the local entries occur, and
  // A.row_offsets_external will describe where the external entries occur.

  typedef typename MatrixType::ScalarType Scalar;

  size_t nrows = A.rows.size();
  std::vector<int> tmp_row_offsets(nrows * 2);
  std::vector<int> tmp_row_offsets_external(nrows * 2);

  int num_local_nz = 0;
  int num_extern_nz = 0;

  // First sort within each row of A, so that local entries come
  // before external entries within each row.
  // tmp_row_offsets describe the locations of the local entries, and
  // tmp_row_offsets_external describe the locations of the external entries.
  //
  for (size_t i = 0; i < nrows; ++i) {
    int *row_begin = &A.packed_cols[static_cast<size_t>(A.row_offsets[i])];
    int *row_end = &A.packed_cols[static_cast<size_t>(A.row_offsets[i + 1])];

    Scalar *coef_row_begin =
        &A.packed_coefs[static_cast<size_t>(A.row_offsets[i])];

    tmp_row_offsets[i * 2] = A.row_offsets[i];
    tmp_row_offsets[i * 2 + 1] = A.row_offsets[i + 1];
    tmp_row_offsets_external[i * 2] = A.row_offsets[i + 1];
    tmp_row_offsets_external[i * 2 + 1] = A.row_offsets[i + 1];

    ptrdiff_t row_len = row_end - row_begin;

    sort_with_companions(row_len, row_begin, coef_row_begin);

    int *row_iter = std::lower_bound(row_begin, row_end, nrows);

    int offset = static_cast<int>(A.row_offsets[i] + row_iter - row_begin);
    tmp_row_offsets[i * 2 + 1] = offset;
    tmp_row_offsets_external[i * 2] = offset;

    num_local_nz += tmp_row_offsets[i * 2 + 1] - tmp_row_offsets[i * 2];
    num_extern_nz +=
        tmp_row_offsets_external[i * 2 + 1] - tmp_row_offsets_external[i * 2];
  }

  // Next, copy the external entries into separate arrays.

  std::vector<int> ext_cols(static_cast<size_t>(num_extern_nz));
  std::vector<Scalar> ext_coefs(static_cast<size_t>(num_extern_nz));
  std::vector<int> ext_offsets(nrows + 1);
  int offset = 0;
  for (size_t i = 0; i < nrows; ++i) {
    ext_offsets[i] = offset;
    for (int j = tmp_row_offsets_external[i * 2];
         j < tmp_row_offsets_external[i * 2 + 1]; ++j) {
      ext_cols[static_cast<size_t>(offset)] =
          A.packed_cols[static_cast<size_t>(j)];
      ext_coefs[static_cast<size_t>(offset++)] =
          A.packed_coefs[static_cast<size_t>(j)];
    }
  }
  ext_offsets[nrows] = offset;

  // Now slide all local entries down to the beginning of A's packed arrays

  A.row_offsets.resize(nrows + 1);
  offset = 0;
  for (size_t i = 0; i < nrows; ++i) {
    A.row_offsets[i] = offset;
    for (int j = tmp_row_offsets[i * 2]; j < tmp_row_offsets[i * 2 + 1]; ++j) {
      A.packed_cols[static_cast<size_t>(offset)] =
          A.packed_cols[static_cast<size_t>(j)];
      A.packed_coefs[static_cast<size_t>(offset++)] =
          A.packed_coefs[static_cast<size_t>(j)];
    }
  }
  A.row_offsets[nrows] = offset;

  // Finally, copy the external entries back into A.packed_cols and
  // A.packed_coefs, starting at the end of the local entries.

  for (int i = offset; i < offset + static_cast<int>(ext_cols.size()); ++i) {
    A.packed_cols[static_cast<size_t>(i)] =
        ext_cols[static_cast<size_t>(i - offset)];
    A.packed_coefs[static_cast<size_t>(i)] =
        ext_coefs[static_cast<size_t>(i - offset)];
  }

  A.row_offsets_external.resize(nrows + 1);
  for (size_t i = 0; i <= nrows; ++i)
    A.row_offsets_external[i] = ext_offsets[i] + offset;
}

//------------------------------------------------------------------------
template <typename MatrixType>
void zero_row_and_put_1_on_diagonal(MatrixType &A, int row) {
  typedef typename MatrixType::ScalarType Scalar;

  size_t row_len = 0;
  int *cols = NULL;
  Scalar *coefs = NULL;
  A.get_row_pointers(row, row_len, cols, coefs);

  for (size_t i = 0; i < row_len; ++i) {
    if (cols[i] == row)
      coefs[i] = 1;
    else
      coefs[i] = 0;
  }
}

//------------------------------------------------------------------------
template <typename MatrixType, typename VectorType>
void impose_dirichlet(typename MatrixType::ScalarType prescribed_value,
                      MatrixType &A, VectorType &b,
                      const std::set<int> &bc_rows) {
  typedef typename MatrixType::ScalarType Scalar;

  int first_local_row = A.rows.size() > 0 ? A.rows[0] : 0;
  int last_local_row = A.rows.size() > 0 ? A.rows[A.rows.size() - 1] : -1;

  typename std::set<int>::const_iterator bc_iter = bc_rows.begin(),
                                         bc_end = bc_rows.end();
  for (; bc_iter != bc_end; ++bc_iter) {
    int row = *bc_iter;
    if (row >= first_local_row && row <= last_local_row) {
      size_t local_row = static_cast<size_t>(row - first_local_row);
      b.coefs[local_row] = prescribed_value;
      zero_row_and_put_1_on_diagonal(A, row);
    }
  }

  for (size_t i = 0; i < A.rows.size(); ++i) {
    int row = A.rows[i];

    if (bc_rows.find(row) != bc_rows.end())
      continue;

    size_t row_length = 0;
    int *cols = NULL;
    Scalar *coefs = NULL;
    A.get_row_pointers(row, row_length, cols, coefs);

    Scalar sum = 0;
    for (size_t j = 0; j < row_length; ++j) {
      if (bc_rows.find(cols[j]) != bc_rows.end()) {
        sum += coefs[j];
        coefs[j] = 0;
      }
    }

    b.coefs[i] -= sum * prescribed_value;
  }
}

static timer_type exchtime = 0;

//------------------------------------------------------------------------
// Compute matrix vector product y = A*x and return dot(x,y), where:
//
// A - input matrix
// x - input vector
// y - result vector
//
template <typename MatrixType, typename VectorType>
typename TypeTraits<typename VectorType::ScalarType>::magnitude_type
matvec_and_dot(MatrixType &A, VectorType &x, VectorType &y,
               cusparseHandle_t cusparse_handle, cublasHandle_t cublas_handle,
               cusparseMatDescr_t descr) {
  timer_type t0 = my_timer();
  exchange_externals(A, x);
  timer_type exchange_time = my_timer() - t0;
  exchtime += exchange_time;

  typedef typename TypeTraits<typename VectorType::ScalarType>::magnitude_type
      magnitude;
  typedef typename MatrixType::ScalarType ScalarType;

  size_t n = A.rows.size();
  const int *Arowoffsets = &A.row_offsets[0];
  const int *Acols = &A.packed_cols[0];
  const ScalarType *Acoefs = &A.packed_coefs[0];
  const ScalarType *xcoefs = &x.coefs[0];
  ScalarType *ycoefs = &y.coefs[0];
  magnitude result = 0;

#ifdef USE_CUDA
  result = cuda_matvec_and_dot(
      Acoefs, Arowoffsets, Acols, xcoefs, ycoefs, static_cast<int>(n),
      static_cast<int>(A.num_cols), static_cast<int>(A.num_nonzeros()),
      cusparse_handle, cublas_handle, descr);
#else
#pragma omp parallel for reduction(+ : result)
  for (size_t row = 0; row < n; ++row) {
    ScalarType sum = 0;

    for (int i = Arowoffsets[row]; i < Arowoffsets[row + 1]; ++i) {
      sum += Acoefs[i] * xcoefs[Acols[i]];
    }

    ycoefs[row] = sum;
    result += xcoefs[row] * sum;
  }
#endif

#ifdef HAVE_MPI
  magnitude local_dot = result, global_dot = 0;
  MPI_Datatype mpi_dtype = TypeTraits<magnitude>::mpi_type();
  MPI_Allreduce(&local_dot, &global_dot, 1, mpi_dtype, MPI_SUM, MPI_COMM_WORLD);
  return global_dot;
#else
  return result;
#endif
}

//------------------------------------------------------------------------
// Compute matrix vector product y = A*x where:
//
// A - input matrix
// x - input vector
// y - result vector
//
template <typename MatrixType, typename VectorType> struct matvec_std {
  void operator()(MatrixType &A, VectorType &x, VectorType &y,
                  cusparseHandle_t cusparse_handle, cusparseMatDescr_t descr) {
    exchange_externals(A, x);

    typedef typename MatrixType::ScalarType ScalarType;

    const int *Arowoffsets = &A.row_offsets[0];
    const int *Acols = &A.packed_cols[0];
    const ScalarType *Acoefs = &A.packed_coefs[0];
    const ScalarType *xcoefs = &x.coefs[0];
    ScalarType *ycoefs = &y.coefs[0];

    size_t n = A.rows.size();

#ifdef USE_CUDA
    cuda_matvec(Acoefs, Arowoffsets, Acols, xcoefs, ycoefs, 0,
                static_cast<int>(n), static_cast<int>(A.num_cols),
                static_cast<int>(A.num_nonzeros()), cusparse_handle, descr);
#else
#pragma omp parallel for
    for (size_t row = 0; row < n; ++row) {
      ScalarType sum = 0;

      for (int i = Arowoffsets[row]; i < Arowoffsets[row + 1]; ++i) {
        sum += Acoefs[i] * xcoefs[Acols[i]];
      }

      ycoefs[row] = sum;
    }
#endif
  }
};

template <typename MatrixType, typename VectorType>
void matvec(MatrixType &A, VectorType &x, VectorType &y,
            cusparseHandle_t handle, cusparseMatDescr_t descr) {
  matvec_std<MatrixType, VectorType> mv;
  mv(A, x, y, handle, descr);
}

template <typename MatrixType, typename VectorType> struct matvec_overlap {
  void operator()(MatrixType &A, VectorType &x, VectorType &y,
                  cusparseHandle_t cusparse_handle, cusparseMatDescr_t descr) {
#ifdef HAVE_MPI
    begin_exchange_externals(A, x);
#endif

    typedef typename MatrixType::ScalarType ScalarType;

    size_t n = A.rows.size();
    const int *Arowoffsets = &A.row_offsets[0];
    const int *Acols = &A.packed_cols[0];
    const ScalarType *Acoefs = &A.packed_coefs[0];
    const ScalarType *xcoefs = &x.coefs[0];
    ScalarType *ycoefs = &y.coefs[0];
    ScalarType beta = 0;

#ifdef USE_CUDA
    cuda_matvec(Acoefs, Arowoffsets, Acols, xcoefs, ycoefs, beta,
                static_cast<int>(n), static_cast<int>(A.num_cols),
                static_cast<int>(A.num_nonzeros()), cusparse_handle, descr);
#else
    for (size_t row = 0; row < n; ++row) {
      ScalarType sum = beta * ycoefs[row];

#pragma omp parallel for
      for (int i = Arowoffsets[row]; i < Arowoffsets[row + 1]; ++i) {
        sum += Acoefs[i] * xcoefs[Acols[i]];
      }

      ycoefs[row] = sum;
    }
#endif

#ifdef HAVE_MPI
    finish_exchange_externals(static_cast<int>(A.neighbors.size()));

    Arowoffsets = &A.row_offsets_external[0];
    beta = 1;

#ifdef USE_CUDA
    cuda_matvec(Acoefs, Arowoffsets, Acols, xcoefs, ycoefs, beta,
                static_cast<int>(n), static_cast<int>(A.num_cols),
                static_cast<int>(A.num_nonzeros()), cusparse_handle, descr);
#else
    for (size_t row = 0; row < n; ++row) {
      ScalarType sum = beta * ycoefs[row];

#pragma omp parallel for
      for (int i = Arowoffsets[row]; i < Arowoffsets[row + 1]; ++i) {
        sum += Acoefs[i] * xcoefs[Acols[i]];
      }

      ycoefs[row] = sum;
    }
#endif

#endif
  }
};

} // namespace miniFE
