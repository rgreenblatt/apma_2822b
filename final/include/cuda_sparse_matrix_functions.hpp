#pragma once

#include "cuda_utils.hpp"
#include "cuda_vector_functions.hpp"
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <iostream>

namespace miniFE {

void cuda_matvec(const double *Acoefs, const int *Arowoffsets, const int *Acols,
                 const double *xcoefs, double *ycoefs, double beta, int n_rows,
                 int n_cols, int nnz, cusparseHandle_t cusparse_handle,
                 cusparseMatDescr_t descr) {

  double alpha = 1.0;

  cusparse_error_chk(cusparseDcsrmv(
      cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_rows, n_cols, nnz,
      &alpha, descr, Acoefs, Arowoffsets, Acols, xcoefs, &beta, ycoefs));
}

void cuda_matvec(const float *Acoefs, const int *Arowoffsets, const int *Acols,
                 const float *xcoefs, float *ycoefs, float beta, int n_rows,
                 int n_cols, int nnz, cusparseHandle_t cusparse_handle,
                 cusparseMatDescr_t descr) {

  float alpha = 1.0;

  cusparse_error_chk(cusparseScsrmv(
      cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_rows, n_cols, nnz,
      &alpha, descr, Acoefs, Arowoffsets, Acols, xcoefs, &beta, ycoefs));
}

double cuda_matvec_and_dot(const double *Acoefs, const int *Arowoffsets,
                           const int *Acols, const double *xcoefs,
                           double *ycoefs, int n_rows, int n_cols, int nnz,
                           cusparseHandle_t cusparse_handle,
                           cublasHandle_t cublas_handle,
                           cusparseMatDescr_t descr) {
  cuda_matvec(Acoefs, Arowoffsets, Acols, xcoefs, ycoefs, 0, n_rows, n_cols, nnz,
              cusparse_handle, descr);
  double result;

  cuda_dot(cublas_handle, xcoefs, ycoefs, &result, n_rows);
  return result;
}

float cuda_matvec_and_dot(const float *Acoefs, const int *Arowoffsets,
                          const int *Acols, const float *xcoefs, float *ycoefs,
                          int n_rows, int n_cols, int nnz,
                          cusparseHandle_t cusparse_handle,
                          cublasHandle_t cublas_handle,
                          cusparseMatDescr_t descr) {
  cuda_matvec(Acoefs, Arowoffsets, Acols, xcoefs, ycoefs, 0, n_rows, n_cols, nnz,
              cusparse_handle, descr);
  float result;

  cuda_dot(cublas_handle, xcoefs, ycoefs, &result, n_rows);
  return result;
}

} // namespace miniFE
