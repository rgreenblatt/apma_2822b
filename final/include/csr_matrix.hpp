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
#include <vector>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "vector_uvm.hpp"

namespace miniFE {

template <typename Scalar>
struct CSRMatrix {
  CSRMatrix()
      : has_local_indices(false), rows(), row_offsets(), row_offsets_external(),
        packed_cols(), packed_coefs(), num_cols(0)
#ifdef HAVE_MPI
        ,
        external_index(), external_local_index(), elements_to_send(),
        neighbors(), recv_length(), send_length(), send_buffer(), request()
#endif
  {
  }

  ~CSRMatrix() {}

  typedef Scalar ScalarType;

  bool has_local_indices;
  AllocVec<int> rows;
  AllocVec<int> row_offsets;
  AllocVec<int> row_offsets_external;
  AllocVec<int> packed_cols;
  AllocVec<Scalar> packed_coefs;
  int num_cols;

#ifdef HAVE_MPI
  std::vector<int> external_index;
  std::vector<int> external_local_index;
  std::vector<int> elements_to_send;
  std::vector<int> neighbors;
  std::vector<int> recv_length;
  std::vector<int> send_length;
  std::vector<Scalar> send_buffer;
  std::vector<MPI_Request> request;
#endif

  int num_nonzeros() const { return row_offsets[row_offsets.size() - 1]; }

  void reserve_space(int nrows, unsigned ncols_per_row) {
    rows.resize(static_cast<size_t>(nrows));
    row_offsets.resize(static_cast<size_t>(nrows) + 1);
    packed_cols.reserve(static_cast<size_t>(nrows) * ncols_per_row);
    packed_coefs.reserve(static_cast<size_t>(nrows) * ncols_per_row);
  }

  void get_row_pointers(int row, size_t &row_length,
                        int *&cols, ScalarType *&coefs) {
    ptrdiff_t local_row = -1;
    // first see if we can get the local-row index using fast direct lookup:
    if (rows.size() >= 1) {
      ptrdiff_t idx = row - rows[0];
      if (idx < static_cast<ptrdiff_t>(rows.size()) && rows[static_cast<size_t>(idx)] == row) {
        local_row = idx;
      }
    }

    // if we didn't get the local-row index using direct lookup, try a
    // more expensive binary-search:
    if (local_row == -1) {
      auto row_iter = std::lower_bound(rows.begin(), rows.end(), row);

      // if we still haven't found row, it's not local so jump out:
      if (row_iter == rows.end() || *row_iter != row) {
        row_length = 0;
        return;
      }

      local_row = row_iter - rows.begin();
    }

    int offset = row_offsets[static_cast<size_t>(local_row)];
    row_length = static_cast<size_t>(
        row_offsets[static_cast<size_t>(local_row) + 1] - offset);
    cols = &packed_cols[static_cast<size_t>(offset)];
    coefs = &packed_coefs[static_cast<size_t>(offset)];
  }
};

} // namespace miniFE
