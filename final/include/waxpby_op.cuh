#pragma once

#include <cstdio>

template <typename ScalarType>
inline
#ifdef __CUDACC__
    __host__ __device__
#endif
    void
    waxpby_op(ScalarType *wcoefs, ScalarType alpha, const ScalarType *xcoefs,
              ScalarType beta, const ScalarType *ycoefs, size_t i) {
  wcoefs[i] = alpha * xcoefs[i] + beta * ycoefs[i];
}
