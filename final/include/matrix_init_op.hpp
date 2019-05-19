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

#include "box_utils.hpp"
#include "csr_matrix.hpp"
#include "simple_mesh_description.hpp"

#include <algorithm>

void sort_if_needed(int *list, int list_len) {
  bool need_to_sort = false;
  for (int i = list_len - 1; i >= 1; --i) {
    if (list[i] < list[i - 1]) {
      need_to_sort = true;
      break;
    }
  }

  if (need_to_sort) {
    std::sort(list, list + list_len);
  }
}

template <typename MatrixType> struct MatrixInitOp {};

template <>
struct MatrixInitOp<miniFE::CSRMatrix<MINIFE_SCALAR>> {
  typedef MINIFE_SCALAR ScalarType;

  const int *rows;
  const int *row_offsets;
  const int *row_coords;

  int global_nodes_x;
  int global_nodes_y;
  int global_nodes_z;

  int global_nrows;

  int *dest_rows;
  int *dest_rowoffsets;
  int *dest_cols;
  ScalarType *dest_coefs;
  size_t n;

  const miniFE::simple_mesh_description *mesh;

  MatrixInitOp(
      const std::vector<int> &rows_vec,
      const std::vector<int> &row_offsets_vec,
      const std::vector<int> &row_coords_vec, int global_nx, int global_ny,
      int global_nz, int global_n_rows,
      const miniFE::simple_mesh_description &input_mesh,
      miniFE::CSRMatrix<MINIFE_SCALAR> &matrix)
      : rows(&rows_vec[0]), row_offsets(&row_offsets_vec[0]),
        row_coords(&row_coords_vec[0]), global_nodes_x(global_nx),
        global_nodes_y(global_ny), global_nodes_z(global_nz),
        global_nrows(global_n_rows), dest_rows(&matrix.rows[0]),
        dest_rowoffsets(&matrix.row_offsets[0]),
        dest_cols(&matrix.packed_cols[0]), dest_coefs(&matrix.packed_coefs[0]),
        n(matrix.rows.size()), mesh(&input_mesh) {
    if (matrix.packed_cols.capacity() != matrix.packed_coefs.capacity()) {
      std::cout << "Warning, packed_cols.capacity ("
                << matrix.packed_cols.capacity() << ") != "
                << "packed_coefs.capacity (" << matrix.packed_coefs.capacity()
                << ")" << std::endl;
    }

    int nnz = row_offsets_vec[n];
    if (static_cast<int>(matrix.packed_cols.capacity()) <
        nnz) {
      std::cout << "Warning, packed_cols.capacity ("
                << matrix.packed_cols.capacity()
                << ") < "
                   " nnz ("
                << nnz << ")" << std::endl;
    }

    matrix.packed_cols.resize(static_cast<size_t>(nnz));
    matrix.packed_coefs.resize(static_cast<size_t>(nnz));
    dest_rowoffsets[n] = nnz;
  }

  inline void operator()(int i) {
    dest_rows[i] = rows[i];
    int offset = row_offsets[i];
    dest_rowoffsets[i] = offset;
    int ix = row_coords[i * 3];
    int iy = row_coords[i * 3 + 1];
    int iz = row_coords[i * 3 + 2];
    int nnz = 0;
    for (int sz = -1; sz <= 1; ++sz) {
      for (int sy = -1; sy <= 1; ++sy) {
        for (int sx = -1; sx <= 1; ++sx) {
          int col_id = miniFE::get_id(
              global_nodes_x, global_nodes_y, global_nodes_z, ix + sx, iy + sy,
              iz + sz);
          if (col_id >= 0 && col_id < global_nrows) {
            int col = mesh->map_id_to_row(col_id);
            dest_cols[offset + nnz] = col;
            dest_coefs[offset + nnz] = 0;
            ++nnz;
          }
        }
      }
    }

    sort_if_needed(&dest_cols[offset], nnz);
  }
};
