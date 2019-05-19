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

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "cuda_utils.cuh"
#include "cusparse.h"
#include "my_timer.hpp"
#include "sparse_matrix_functions.hpp"
#include "vector_functions.hpp"

#include "outstream.hpp"

namespace miniFE {

template <typename VectorType>
bool breakdown(typename VectorType::ScalarType inner, const VectorType &v,
               const VectorType &w, cublasHandle_t cublas_handle) {
  typedef typename VectorType::ScalarType Scalar;
  typedef typename TypeTraits<Scalar>::magnitude_type magnitude;

  // This is code that was copied from Aztec, and originally written
  // by my hero, Ray Tuminaro.
  //
  // Assuming that inner = <v,w> (inner product of v and w),
  // v and w are considered orthogonal if
  //  |inner| < 100 * ||v||_2 * ||w||_2 * epsilon

  magnitude vnorm = std::sqrt(dot(v, v, cublas_handle));
  magnitude wnorm = std::sqrt(dot(w, w, cublas_handle));
  return std::abs(inner) <=
         100 * vnorm * wnorm * std::numeric_limits<magnitude>::epsilon();
}

template <typename OperatorType, typename VectorType, typename Matvec>
void cg_solve(
    OperatorType &A, const VectorType &b, VectorType &x, Matvec matvec,
    int max_iter,
    typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type
        &tolerance,
    int &num_iters,
    typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type
        &normr,
    timer_type *my_cg_times) {
  typedef typename OperatorType::ScalarType ScalarType;
  typedef typename TypeTraits<ScalarType>::magnitude_type magnitude_type;

  timer_type t0 = 0, tWAXPY = 0, tDOT = 0, tMATVEC = 0, tMATVECDOT = 0;
  timer_type total_time = my_timer();

  int myproc = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  if (!A.has_local_indices) {
    std::cerr << "miniFE::cg_solve ERROR, A.has_local_indices is false, needs "
                 "to be true. This probably means "
              << "miniFE::make_local_matrix(A) was not called prior to calling "
                 "miniFE::cg_solve."
              << std::endl;
    return;
  }

  int nrows = static_cast<int>(A.rows.size());
  int ncols = A.num_cols;

  VectorType r(b.startIndex, nrows);
  VectorType p(0, ncols);
  VectorType Ap(b.startIndex, nrows);

  cublasHandle_t cublas_handle = 0;
  cusparseHandle_t cusparse_handle = 0;
  cusparseMatDescr_t descr = 0;
#ifdef USE_CUDA
  cublas_error_chk(cublasCreate_v2(&cublas_handle));
  cusparse_error_chk(cusparseCreate(&cusparse_handle));
  cusparse_error_chk(cusparseCreateMatDescr(&descr));
  cusparse_error_chk(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  cusparse_error_chk(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
#endif

  normr = 0;
  magnitude_type rtrans = 0;
  magnitude_type oldrtrans = 0;

  int print_freq = max_iter / 10;
  if (print_freq > 50) {
    print_freq = 50;
  }
  if (print_freq < 1) {
    print_freq = 1;
  }

  ScalarType one = 1.0;
  ScalarType zero = 0.0;

  TICK();
  waxpby(one, x, zero, x, p);
  TOCK(tWAXPY);

  TICK();
  matvec(A, p, Ap, cusparse_handle, descr);
  TOCK(tMATVEC);

  TICK();
  waxpby(one, b, -one, Ap, r);
  TOCK(tWAXPY);

  TICK();
  rtrans = dot(r, r, cublas_handle);
  TOCK(tDOT);

  normr = std::sqrt(rtrans);

  if (myproc == 0) {
    std::cout << "Initial Residual = " << normr << std::endl;
  }

  magnitude_type brkdown_tol = std::numeric_limits<magnitude_type>::epsilon();

#ifdef MINIFE_DEBUG
  std::ostream &os = outstream();
  os << "brkdown_tol = " << brkdown_tol << std::endl;
#endif

  for (int k = 1; k <= max_iter && normr > tolerance; ++k) {
    if (k == 1) {
      TICK();

      waxpby(one, r, zero, r, p);

      TOCK(tWAXPY);
    } else {
      oldrtrans = rtrans;
      TICK();
      rtrans = dot(r, r, cublas_handle);
      TOCK(tDOT);
      magnitude_type beta = rtrans / oldrtrans;
      TICK();
      waxpby(one, r, beta, p, p);
      TOCK(tWAXPY);
    }

    normr = std::sqrt(rtrans);

    if (myproc == 0 && (k % print_freq == 0 || k == max_iter)) {
      std::cout << "Iteration = " << k << "   Residual = " << normr
                << std::endl;
    }

    magnitude_type alpha = 0;
    magnitude_type p_ap_dot = 0;

#ifdef MINIFE_FUSED
    TICK();
    p_ap_dot = matvec_and_dot(A, p, Ap, cusparse_handle, cublas_handle, descr);
    TOCK(tMATVECDOT);
#else
    TICK();
    matvec(A, p, Ap, cusparse_handle, descr);
    TOCK(tMATVEC);

    TICK();
    p_ap_dot = dot(Ap, p, cublas_handle);
    TOCK(tDOT);
#endif

#ifdef MINIFE_DEBUG
    os << "iter " << k << ", p_ap_dot = " << p_ap_dot;
    os.flush();
#endif
    if (p_ap_dot < brkdown_tol) {
      if (p_ap_dot < 0 || breakdown(p_ap_dot, Ap, p, cublas_handle)) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"
                  << std::endl;
#ifdef MINIFE_DEBUG
        os << "ERROR, numerical breakdown!" << std::endl;
#endif
        // update the timers before jumping out.
        my_cg_times[WAXPY] = tWAXPY;
        my_cg_times[DOT] = tDOT;
        my_cg_times[MATVEC] = tMATVEC;
        my_cg_times[TOTAL] = my_timer() - total_time;
        return;
      } else
        brkdown_tol = 0.1 * p_ap_dot;
    }
    alpha = rtrans / p_ap_dot;
#ifdef MINIFE_DEBUG
    os << ", rtrans = " << rtrans << ", alpha = " << alpha << std::endl;
#endif

#ifdef MINIFE_FUSED
    TICK();
    fused_waxpby(one, x, alpha, p, x, one, r, -alpha, Ap, r);
    TOCK(tWAXPY);
#else
    TICK();
    waxpby(one, x, alpha, p, x);
    waxpby(one, r, -alpha, Ap, r);
    TOCK(tWAXPY);
#endif

    num_iters = k;
  }

#ifdef USE_CUDA
  cublas_error_chk(cublasDestroy_v2(cublas_handle));
  cusparse_error_chk(cusparseDestroy(cusparse_handle));
#endif

  my_cg_times[WAXPY] = tWAXPY;
  my_cg_times[DOT] = tDOT;
  my_cg_times[MATVEC] = tMATVEC;
  my_cg_times[MATVECDOT] = tMATVECDOT;
  my_cg_times[TOTAL] = my_timer() - total_time;
}

} // namespace miniFE
