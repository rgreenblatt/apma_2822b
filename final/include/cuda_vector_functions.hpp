#pragma once

#include <cstdio>

template <typename ScalarType>
void cuda_waxpby_fused(ScalarType *wcoefs, ScalarType *w2coefs,
                       ScalarType alpha, ScalarType alpha2,
                       const ScalarType *xcoefs, const ScalarType *x2coefs,
                       ScalarType beta, ScalarType beta2,
                       const ScalarType *ycoefs, const ScalarType *y2coefs,
                       unsigned n);
