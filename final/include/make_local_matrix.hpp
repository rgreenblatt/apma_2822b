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

#include <map>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace miniFE {

// The following function was converted from Mike's HPCCG code.
template <typename MatrixType> void make_local_matrix(MatrixType &A) {
#ifdef HAVE_MPI
  int numprocs = 1, myproc = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

  typedef typename MatrixType::ScalarType Scalar;

  if (numprocs < 2) {
    A.num_cols = static_cast<int>(A.rows.size());
    A.has_local_indices = true;
    return;
  }

  std::map<int, int> externals;
  int num_external = 0;

  // Extract Matrix pieces

  int local_nrow = static_cast<int>(A.rows.size());
  int start_row = local_nrow > 0 ? A.rows[0] : -1;
  int stop_row =
      local_nrow > 0 ? A.rows[static_cast<size_t>(local_nrow) - 1] : -1;

  // We need to convert the index values for the rows on this processor
  // to a local index space. We need to:
  // - Determine if each index reaches to a local value or external value
  // - If local, subtract start_row from index value to get local index
  // - If external, find out if it is already accounted for.
  //   - If so, then do nothing,
  //   - otherwise
  //     - add it to the list of external indices,
  //     - find out which processor owns the value.
  //     - Set up communication for sparse MV operation

  ///////////////////////////////////////////
  // Scan the indices and transform to local
  ///////////////////////////////////////////

  std::vector<int> &external_index = A.external_index;

  for (size_t i = 0; i < A.rows.size(); ++i) {
    int *Acols = NULL;
    Scalar *Acoefs = NULL;
    size_t row_len = 0;
    A.get_row_pointers(A.rows[i], row_len, Acols, Acoefs);

    for (size_t j = 0; j < row_len; ++j) {
      int cur_ind = Acols[j];
      if (start_row <= cur_ind && cur_ind <= stop_row) {
        Acols[j] -= start_row;
      } else { // Must find out if we have already set up this point
        if (externals.find(cur_ind) == externals.end()) {
          externals[cur_ind] = num_external++;
          external_index.push_back(cur_ind);
        }
        // Mark index as external by adding 1 and negating it
        Acols[j] = -(Acols[j] + 1);
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////
  // Go through list of externals to find out which processors must be accessed.
  ////////////////////////////////////////////////////////////////////////

  std::vector<int> tmp_buffer(static_cast<size_t>(numprocs),
                                        0); // Temp buffer space needed below

  // Build list of global index offset

  std::vector<int> global_index_offsets(static_cast<size_t>(numprocs),
                                                  0);

  tmp_buffer[static_cast<size_t>(myproc)] = start_row; // This is my start row

  // This call sends the start_row of each ith processor to the ith
  // entry of global_index_offsets on all processors.
  // Thus, each processor knows the range of indices owned by all
  // other processors.
  // Note: There might be a better algorithm for doing this, but this
  //       will work...

  MPI_Datatype mpi_dtype = TypeTraits<int>::mpi_type();
  MPI_Allreduce(&tmp_buffer[0], &global_index_offsets[0], numprocs, mpi_dtype,
                MPI_SUM, MPI_COMM_WORLD);

  // Go through list of externals and find the processor that owns each
  std::vector<int> external_processor(static_cast<size_t>(num_external));

  for (int i = 0; i < num_external; ++i) {
    int cur_ind = external_index[static_cast<size_t>(i)];
    for (int j = numprocs - 1; j >= 0; --j) {
      if (global_index_offsets[static_cast<size_t>(j)] <= cur_ind &&
          global_index_offsets[static_cast<size_t>(j)] >= 0) {
        external_processor[static_cast<size_t>(i)] = j;
        break;
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////
  // Sift through the external elements. For each newly encountered external
  // point assign it the next index in the sequence. Then look for other
  // external elements who are updated by the same node and assign them the next
  // set of index numbers in the sequence (ie. elements updated by the same node
  // have consecutive indices).
  /////////////////////////////////////////////////////////////////////////

  int count = local_nrow;
  std::vector<int> &external_local_index = A.external_local_index;
  external_local_index.assign(static_cast<size_t>(num_external), -1);

  for (int i = 0; i < num_external; ++i) {
    if (external_local_index[static_cast<size_t>(i)] == -1) {
      external_local_index[static_cast<size_t>(i)] = count++;

      for (int j = i + 1; j < num_external; ++j) {
        if (external_processor[static_cast<size_t>(j)] ==
            external_processor[static_cast<size_t>(i)])
          external_local_index[static_cast<size_t>(j)] = count++;
      }
    }
  }

  for (size_t i = 0; i < static_cast<size_t>(local_nrow); ++i) {
    int *Acols = NULL;
    Scalar *Acoefs = NULL;
    size_t row_len = 0;
    A.get_row_pointers(A.rows[i], row_len, Acols, Acoefs);

    for (size_t j = 0; j < row_len; ++j) {
      if (Acols[j] < 0) { // Change index values of externals
        int cur_ind = -Acols[j] - 1;
        Acols[j] =
            external_local_index[static_cast<size_t>(externals[cur_ind])];
      }
    }
  }

  std::vector<int> new_external_processor(static_cast<size_t>(num_external), 0);

  for (int i = 0; i < num_external; ++i) {
    new_external_processor[static_cast<size_t>(
        external_local_index[static_cast<size_t>(i)] - local_nrow)] =
        external_processor[static_cast<size_t>(i)];
  }

  ////////////////////////////////////////////////////////////////////////
  ///
  // Count the number of neighbors from which we receive information to update
  // our external elements. Additionally, fill the array tmp_neighbors in the
  // following way:
  //      tmp_neighbors[i] = 0   ==>  No external elements are updated by
  //                              processor i.
  //      tmp_neighbors[i] = x   ==>  (x-1)/numprocs elements are updated from
  //                              processor i.
  ///
  ////////////////////////////////////////////////////////////////////////

  std::vector<int> tmp_neighbors(static_cast<size_t>(numprocs), 0);

  int num_recv_neighbors = 0;
  int length = 1;

  for (int i = 0; i < num_external; ++i) {
    if (tmp_neighbors[static_cast<size_t>(
            new_external_processor[static_cast<size_t>(i)])] == 0) {
      ++num_recv_neighbors;
      tmp_neighbors[static_cast<size_t>(
          new_external_processor[static_cast<size_t>(i)])] = 1;
    }
    tmp_neighbors[static_cast<size_t>(
        new_external_processor[static_cast<size_t>(i)])] += numprocs;
  }

  /// sum over all processor all the tmp_neighbors arrays ///

  MPI_Allreduce(&tmp_neighbors[0], &tmp_buffer[0], numprocs, mpi_dtype, MPI_SUM,
                MPI_COMM_WORLD);

  // decode the combined 'tmp_neighbors' (stored in tmp_buffer)
  // array from all the processors

  int num_send_neighbors =
      tmp_buffer[static_cast<size_t>(myproc)] % numprocs;

  /// decode 'tmp_buffer[myproc] to deduce total number of elements
  //  we must send

  int total_to_be_sent =
      (tmp_buffer[static_cast<size_t>(myproc)] - num_send_neighbors) / numprocs;

  ///////////////////////////////////////////////////////////////////////
  ///
  // Make a list of the neighbors that will send information to update our
  // external elements (in the order that we will receive this information).
  ///
  ///////////////////////////////////////////////////////////////////////

  std::vector<int> recv_list;
  recv_list.push_back(new_external_processor[0]);
  for (int i = 1; i < num_external; ++i) {
    if (new_external_processor[static_cast<size_t>(i) - 1] !=
        new_external_processor[static_cast<size_t>(i)]) {
      recv_list.push_back(new_external_processor[static_cast<size_t>(i)]);
    }
  }

  //
  // Send a 0 length message to each of our recv neighbors
  //

  std::vector<int> send_list(static_cast<size_t>(num_send_neighbors), 0);

  //
  // first post receives, these are immediate receives
  // Do not wait for result to come, will do that at the
  // wait call below.
  //
  int MPI_MY_TAG = 99;

  std::vector<MPI_Request> request(static_cast<size_t>(num_send_neighbors));
  for (int i = 0; i < num_send_neighbors; ++i) {
    MPI_Irecv(&tmp_buffer[static_cast<size_t>(i)], 1, mpi_dtype, MPI_ANY_SOURCE,
              MPI_MY_TAG, MPI_COMM_WORLD, &request[static_cast<size_t>(i)]);
  }

  // send messages

  for (int i = 0; i < num_recv_neighbors; ++i) {
    MPI_Send(&tmp_buffer[static_cast<size_t>(i)], 1, mpi_dtype,
             recv_list[static_cast<size_t>(i)], MPI_MY_TAG, MPI_COMM_WORLD);
  }

  ///
  // Receive message from each send neighbor to construct 'send_list'.
  ///

  MPI_Status status;
  for (int i = 0; i < num_send_neighbors; ++i) {
    if (MPI_Wait(&request[static_cast<size_t>(i)], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    send_list[static_cast<size_t>(i)] = status.MPI_SOURCE;
  }

  //////////////////////////////////////////////////////////////////////
  ///
  // Compare the two lists. In most cases they should be the same.
  // However, if they are not then add new entries to the recv list
  // that are in the send list (but not already in the recv list).
  ///
  //////////////////////////////////////////////////////////////////////

  for (int j = 0; j < num_send_neighbors; ++j) {
    int found = 0;
    for (int i = 0; i < num_recv_neighbors; ++i) {
      if (recv_list[static_cast<size_t>(i)] ==
          send_list[static_cast<size_t>(j)])
        found = 1;
    }

    if (found == 0) {
      recv_list.push_back(send_list[static_cast<size_t>(j)]);
      ++num_recv_neighbors;
    }
  }

  num_send_neighbors = num_recv_neighbors;
  request.resize(static_cast<size_t>(num_send_neighbors));

  A.elements_to_send.assign(static_cast<size_t>(total_to_be_sent), 0);
  A.send_buffer.assign(static_cast<size_t>(total_to_be_sent), 0);

  //
  // Create 'new_external' which explicitly put the external elements in the
  // order given by 'external_local_index'
  //

  std::vector<int> new_external(static_cast<size_t>(num_external));
  for (int i = 0; i < num_external; ++i) {
    new_external[static_cast<size_t>(
        external_local_index[static_cast<size_t>(i)] - local_nrow)] =
        external_index[static_cast<size_t>(i)];
  }

  /////////////////////////////////////////////////////////////////////////
  //
  // Send each processor the global index list of the external elements in the
  // order that I will want to receive them when updating my external elements.
  //
  /////////////////////////////////////////////////////////////////////////

  std::vector<int> lengths(static_cast<size_t>(num_recv_neighbors));

  ++MPI_MY_TAG;

  // First post receives

  for (int i = 0; i < num_recv_neighbors; ++i) {
    int partner = recv_list[static_cast<size_t>(i)];
    MPI_Irecv(&lengths[static_cast<size_t>(i)], 1, MPI_INT, partner, MPI_MY_TAG,
              MPI_COMM_WORLD, &request[static_cast<size_t>(i)]);
  }

  std::vector<int> &neighbors = A.neighbors;
  std::vector<int> &recv_length = A.recv_length;
  std::vector<int> &send_length = A.send_length;

  neighbors.resize(static_cast<size_t>(num_recv_neighbors), 0);
  A.request.resize(static_cast<size_t>(num_recv_neighbors));
  recv_length.resize(static_cast<size_t>(num_recv_neighbors), 0);
  send_length.resize(static_cast<size_t>(num_recv_neighbors), 0);

  int j = 0;
  for (int i = 0; i < num_recv_neighbors; ++i) {
    int start = j;
    int newlength = 0;

    // go through list of external elements until updating
    // processor changes

    while ((j < num_external) &&
           (new_external_processor[static_cast<size_t>(j)] ==
            recv_list[static_cast<size_t>(i)])) {
      ++newlength;
      ++j;
      if (j == num_external)
        break;
    }

    recv_length[static_cast<size_t>(i)] = newlength;
    neighbors[static_cast<size_t>(i)] = recv_list[static_cast<size_t>(i)];

    length = j - start;
    MPI_Send(&length, 1, MPI_INT, recv_list[static_cast<size_t>(i)], MPI_MY_TAG,
             MPI_COMM_WORLD);
  }

  // Complete the receives of the number of externals

  for (int i = 0; i < num_recv_neighbors; ++i) {
    if (MPI_Wait(&request[static_cast<size_t>(i)], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    send_length[static_cast<size_t>(i)] = lengths[static_cast<size_t>(i)];
  }

  ////////////////////////////////////////////////////////////////////////
  // Build "elements_to_send" list. These are the x elements I own
  // that need to be sent to other processors.
  ////////////////////////////////////////////////////////////////////////

  ++MPI_MY_TAG;

  j = 0;
  for (int i = 0; i < num_recv_neighbors; ++i) {
    MPI_Irecv(&A.elements_to_send[static_cast<size_t>(j)],
              send_length[static_cast<size_t>(i)], mpi_dtype,
              neighbors[static_cast<size_t>(i)], MPI_MY_TAG, MPI_COMM_WORLD,
              &request[static_cast<size_t>(i)]);
    j += send_length[static_cast<size_t>(i)];
  }

  j = 0;
  for (int i = 0; i < num_recv_neighbors; ++i) {
    int start = j;
    int newlength = 0;

    // Go through list of external elements
    // until updating processor changes. This is redundant, but
    // saves us from recording this information.

    while ((j < num_external) &&
           (new_external_processor[static_cast<size_t>(j)] ==
            recv_list[static_cast<size_t>(i)])) {
      ++newlength;
      ++j;
      if (j == num_external)
        break;
    }
    MPI_Send(&new_external[static_cast<size_t>(start)], j - start, mpi_dtype,
             recv_list[static_cast<size_t>(i)], MPI_MY_TAG, MPI_COMM_WORLD);
  }

  // receive from each neighbor the global index list of external elements

  for (int i = 0; i < num_recv_neighbors; ++i) {
    if (MPI_Wait(&request[static_cast<size_t>(i)], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  /// replace global indices by local indices ///

  for (int i = 0; i < total_to_be_sent; ++i) {
    A.elements_to_send[static_cast<size_t>(i)] -= start_row;
  }

  //////////////////
  // Finish up !!
  //////////////////

  A.num_cols = static_cast<int>(local_nrow + num_external);

#else
  A.num_cols = static_cast<int>(A.rows.size());
#endif

  A.has_local_indices = true;
}

} // namespace miniFE
