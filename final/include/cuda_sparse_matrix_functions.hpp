#pragma once

#include "cuda_utils.cuh"
#include "cuda_vector_functions.hpp"
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <iostream>

namespace miniFE {

template <typename GlobalOrdinalType, typename LocalOrdinalType>
void cuda_matvec(const double *Acoefs, const LocalOrdinalType *Arowoffsets,
                   const GlobalOrdinalType *Acols, const double *xcoefs,
                   double *ycoefs, int n_rows, int n_cols, int nnz,
                   cusparseHandle_t cusparse_handle, cusparseMatDescr_t descr) {

  double alpha = 1.0;
  double beta = 0.0;

  cusparse_error_chk(cusparseDcsrmv(
      cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_rows, n_cols, nnz, &alpha,
      descr, Acoefs, Arowoffsets, Acols, xcoefs, &beta, ycoefs));
}

template <typename GlobalOrdinalType, typename LocalOrdinalType>
double
cuda_matvec_and_dot(const double *Acoefs, const LocalOrdinalType *Arowoffsets,
                    const GlobalOrdinalType *Acols, const double *xcoefs,
                    double *ycoefs, int n_rows, int n_cols, int nnz,
                    cusparseHandle_t cusparse_handle,
                    cublasHandle_t cublas_handle, cusparseMatDescr_t descr) {
  cuda_matvec(Acoefs, Arowoffsets, Acols, xcoefs, ycoefs, n_rows, n_cols, nnz,
              cusparse_handle, descr);
  double result;

  cuda_dot(cublas_handle, xcoefs, ycoefs, &result, n_rows);
  return result;
}

} // namespace miniFE
