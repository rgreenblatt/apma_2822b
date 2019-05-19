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

#include <fstream>
#include <sstream>
#include <vector>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef MINIFE_HAVE_TBB
#include "locking_vector.hpp"
#endif

#include "dot_op.hpp"
#include "type_traits.hpp"
#include "vector.hpp"
#include "waxpby_op.cuh"

#ifdef USE_CUDA
#include "cuda_vector_functions.hpp"
#endif

namespace miniFE {

template <typename VectorType>
void write_vector(const std::string &filename, const VectorType &vec) {
  int numprocs = 1, myproc = 0;
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  std::ostringstream osstr;
  osstr << filename << "." << numprocs << "." << myproc;
  std::string full_name = osstr.str();
  std::ofstream ofs(full_name.c_str());

  typedef typename VectorType::ScalarType ScalarType;

  const std::vector<ScalarType> &coefs = vec.coefs;
  for (int p = 0; p < numprocs; ++p) {
    if (p == myproc) {
      if (p == 0) {
        ofs << vec.local_size << std::endl;
      }

      typename VectorType::GlobalOrdinalType first = vec.startIndex;
      for (typename VectorType::LocalOrdinalType i = 0; i < vec.local_size;
           ++i) {
        ofs << first + i << " " << coefs[i] << std::endl;
      }
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
}

template <typename VectorType>
void sum_into_vector(size_t num_indices,
                     const typename VectorType::GlobalOrdinalType *indices,
                     const typename VectorType::ScalarType *coefs,
                     VectorType &vec) {
  typedef typename VectorType::GlobalOrdinalType GlobalOrdinal;
  typedef typename VectorType::ScalarType Scalar;

  GlobalOrdinal first = vec.startIndex;
  GlobalOrdinal last = first + vec.local_size - 1;

  AllocVec<Scalar> &vec_coefs = vec.coefs;

  for (size_t i = 0; i < num_indices; ++i) {
    if (indices[i] < first || indices[i] > last)
      continue;
    size_t idx = static_cast<size_t>(indices[i] - first);
    vec_coefs[idx] += coefs[i];
  }
}

#ifdef MINIFE_HAVE_TBB
template <typename VectorType>
void sum_into_vector(size_t num_indices,
                     const typename VectorType::GlobalOrdinalType *indices,
                     const typename VectorType::ScalarType *coefs,
                     LockingVector<VectorType> &vec) {
  vec.sum_in(num_indices, indices, coefs);
}
#endif

//------------------------------------------------------------
// Compute the update of a vector with the sum of two scaled vectors where:
//
// w = alpha*x + beta*y
//
// x,y - input vectors
//
// alpha,beta - scalars applied to x and y respectively
//
// w - output vector
//
template <typename VectorType>
void waxpby(typename VectorType::ScalarType alpha, const VectorType &x,
            typename VectorType::ScalarType beta, const VectorType &y,
            VectorType &w) {
  typedef typename VectorType::ScalarType ScalarType;

#ifdef MINIFE_DEBUG
  if (y.local_size < x.local_size || w.local_size < x.local_size) {
    std::cerr << "miniFE::waxpby ERROR, y and w must be at least as long as x."
              << std::endl;
    return;
  }
#endif

  unsigned n = static_cast<unsigned>(x.coefs.size());
  const ScalarType *xcoefs = &x.coefs[0];
  const ScalarType *ycoefs = &y.coefs[0];
  ScalarType *wcoefs = &w.coefs[0];

#ifdef USE_CUDA
  cuda_waxpby(wcoefs, alpha, xcoefs, beta, ycoefs, n);
#else
  for (unsigned i = 0; i < n; ++i) {
    waxpby_op(wcoefs, alpha, xcoefs, beta, ycoefs, i);
  }
#endif
}

// Like waxpby above, except operates on two sets of arguments.
// In other words, performs two waxpby operations in one loop.
template <typename VectorType>
void fused_waxpby(typename VectorType::ScalarType alpha, const VectorType &x,
                  typename VectorType::ScalarType beta, const VectorType &y,
                  VectorType &w, typename VectorType::ScalarType alpha2,
                  const VectorType &x2, typename VectorType::ScalarType beta2,
                  const VectorType &y2, VectorType &w2) {
  typedef typename VectorType::ScalarType ScalarType;

#ifdef MINIFE_DEBUG
  if (y.local_size < x.local_size || w.local_size < x.local_size) {
    std::cerr << "miniFE::waxpby ERROR, y and w must be at least as long as x."
              << std::endl;
    return;
  }
#endif

  unsigned n = static_cast<unsigned>(x.coefs.size());
  const ScalarType *xcoefs = &x.coefs[0];
  const ScalarType *ycoefs = &y.coefs[0];
  ScalarType *wcoefs = &w.coefs[0];

  const ScalarType *x2coefs = &x2.coefs[0];
  const ScalarType *y2coefs = &y2.coefs[0];
  ScalarType *w2coefs = &w2.coefs[0];

#ifdef USE_CUDA
  cuda_waxpby_fused(wcoefs, w2coefs, alpha, alpha2, xcoefs, x2coefs, beta,
                    beta2, ycoefs, y2coefs, n);
#else
#pragma omp parallel for
  for (unsigned i = 0; i < n; ++i) {
    waxpby_op(wcoefs, alpha, xcoefs, beta, ycoefs, i);
    waxpby_op(w2coefs, alpha2, x2coefs, beta2, y2coefs, i);
  }
#endif
}

//-----------------------------------------------------------
// Compute the dot product of two vectors where:
//
// x,y - input vectors
//
// result - return-value
//
template <typename Vector>
typename TypeTraits<typename Vector::ScalarType>::magnitude_type
dot(const Vector &x, const Vector &y, cublasHandle_t handle=0);

template <typename Vector>
typename TypeTraits<typename Vector::ScalarType>::magnitude_type
dot(const Vector &x, const Vector &y, cublasHandle_t handle) {
  size_t n = x.coefs.size();

#ifdef MINIFE_DEBUG
  if (y.local_size < static_cast<typename Vector::LocalOrdinalType>(n)) {
    std::cerr << "miniFE::dot ERROR, y must be at least as long as x."
              << std::endl;
    n = y.local_size;
  }
#endif

  typedef typename Vector::ScalarType Scalar;
  typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type
      magnitude;

  const Scalar *xcoefs = &x.coefs[0];
  const Scalar *ycoefs = &y.coefs[0];
  magnitude result = 0;


#ifdef USE_CUDA
  cuda_dot(handle, xcoefs, ycoefs, &result, static_cast<unsigned>(n));
#else
#pragma omp parallel for reduction(+ : result)
  for (size_t i = 0; i < n; ++i) {
    result += xcoefs[i] * ycoefs[i];
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

} // namespace miniFE
