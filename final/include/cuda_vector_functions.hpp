#pragma once

#include <cstdio>

#include <cublas_v2.h>

namespace miniFE {
template <typename ScalarType>
void cuda_waxpby_fused(ScalarType *wcoefs, ScalarType *w2coefs,
                       ScalarType alpha, ScalarType alpha2,
                       const ScalarType *xcoefs, const ScalarType *x2coefs,
                       ScalarType beta, ScalarType beta2,
                       const ScalarType *ycoefs, const ScalarType *y2coefs,
                       unsigned n);

template <typename ScalarType>
void cuda_waxpby(ScalarType *wcoefs, ScalarType alpha, const ScalarType *xcoefs,
                 ScalarType beta, const ScalarType *ycoefs, int n);

void cuda_dot(cublasHandle_t handle, const double *xcoefs, const double *ycoefs,
              double *result, int n);
} // namespace miniFE
