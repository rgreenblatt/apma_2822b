#include "cuda_vector_functions.hpp"
#include "waxpby_op.cuh"
#include "cuda_utils.cuh"

template <typename ScalarType>
__global__ void cuda_waxpby_fused_kernel(
    ScalarType *wcoefs, ScalarType *w2coefs, ScalarType alpha,
    ScalarType alpha2, const ScalarType *xcoefs, const ScalarType *x2coefs,
    ScalarType beta, ScalarType beta2, const ScalarType *ycoefs,
    const ScalarType *y2coefs, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    waxpby_op(wcoefs, alpha, xcoefs, beta, ycoefs, i);
    waxpby_op(w2coefs, alpha2, x2coefs, beta2, y2coefs, i);
  }
}

template <typename ScalarType>
void cuda_waxpby_fused(ScalarType *wcoefs, ScalarType *w2coefs, ScalarType alpha,
                  ScalarType alpha2, const ScalarType *xcoefs,
                  const ScalarType *x2coefs, ScalarType beta, ScalarType beta2,
                  const ScalarType *ycoefs, const ScalarType *y2coefs,
                  unsigned n) {

  const unsigned thread_num = 256;
  cuda_waxpby_fused_kernel<<<(n + thread_num - 1) / thread_num, thread_num>>>(wcoefs, w2coefs, alpha, alpha2, xcoefs, x2coefs, beta, beta2, ycoefs,
      y2coefs, n);
  cuda_error_chk(cudaDeviceSynchronize());
}

template void cuda_waxpby_fused(MINIFE_SCALAR *wcoefs, MINIFE_SCALAR *w2coefs,
                                MINIFE_SCALAR alpha, MINIFE_SCALAR alpha2,
                                const MINIFE_SCALAR *xcoefs,
                                const MINIFE_SCALAR *x2coefs,
                                MINIFE_SCALAR beta, MINIFE_SCALAR beta2,
                                const MINIFE_SCALAR *ycoefs,
                                const MINIFE_SCALAR *y2coefs, unsigned n);
