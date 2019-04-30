#include <algorithm> // std::random_shuffle
#include <cfloat>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <utility>
#include <vector> // std::vector

#define USE_NVTX

#ifdef USE_NVTX
#include "nvToolsExt.h"
#endif

#define FULL_MASK 0xffffffff

//primarily for camel case avoidance
enum : uint32_t { warp_size = warpSize, log_warp_size = 5 };

template <typename T> __inline__ __device__ T warp_prefix_sum(T val) {
  T orig_val = val;

  for (uint8_t i = 0;  i < warp_size; ++i) {
    val += __shfl_up_sync(FULL_MASK, val, 1);
  }
  return val - orig_val;
}

template <typename T> __inline__ __device__ T warp_reduce_sum(T val) {
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  return val;
}

#define BITS_PER_PASS 2
#define BINS_PER_PASS 4

__inline__ __host__ __device__ void count_keys(uint32_t *data, uint8_t *counts,
                                               uint32_t start, uint32_t end,
                                               uint8_t shift) {
  for (size_t i = start; i < end; i+=warp_size) {
    // hopefully compiler optimization will save me
    counts[(data[i] >> shift) & (BINS_PER_PASS - 1)]++;
  }
}

__global__ void maxes(uint32_t **data, uint32_t **max_vals, uint32_t **max_locs,
                      uint32_t n, uint32_t m, uint32_t total_count,
                      uint32_t threads_per_n, size_t num_warps_per_n) {

  extern __shared__ uint32_t counts[];

  uint32_t *bin_sums = counts;
  uint32_t *working_data =
      &counts[n * num_warps_per_n * BINS_PER_PASS];
  uint32_t *indexes =
      &counts[n * num_warps_per_n * BINS_PER_PASS + n * m];

  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  size_t n_start = (index * n) / total_count;
  size_t n_end = ((index + 1) * n) / total_count;

  uint32_t m_interval = (m + threads_per_n - 1) / threads_per_n;
  uint32_t m_index = static_cast<uint32_t>(index - (n_start * total_count) / n);

  uint32_t m_start = m_index * m_interval;
  uint32_t m_end = umin((m_index + warp_size) * m_interval, m);

  for (size_t i = n_start; i < n_end; ++i) {
    for (uint32_t j = m_start; j < m_end; j+=warp_size) {
      indexes[i * m + j] = j;
      working_data[i * m + j] = data[n][j];
    }
  }

  auto warp_id = threadIdx.x >> log_warp_size;

  for (uint8_t shift = 0; shift < sizeof(uint32_t); shift += BITS_PER_PASS) {
    for (size_t i = n_start; i < n_end; ++i) {

#define INDEX_BIN_ARRAY(arr, iter_index)                                       \
  (arr[i * threads_per_n * BINS_PER_PASS + iter_index * BINS_PER_PASS + bin])

#define BIN_LOOP(expr)                                                         \
  for (uint8_t bin = 0; bin < BINS_PER_PASS; bin++) {                          \
    expr                                                                       \
  }
      uint8_t counts[BINS_PER_PASS] = {};
      count_keys(&working_data[i], counts, m_start, m_end, shift);

      BIN_LOOP(counts[bin] = warp_reduce_sum(counts[bin]);)

      if (m_index % warp_size == 0) {
        size_t iter_index = (m_index + warp_size - 1) / warp_size;

        BIN_LOOP(INDEX_BIN_ARRAY(bin_sums, iter_index) = counts[bin];)
      }

      __syncthreads();

      //assumption is that num_warps_per_n is less than the warp_size
      if (m_index < num_warps_per_n) {
        BIN_LOOP(INDEX_BIN_ARRAY(bin_sums, m_index) =
                     warp_prefix_sum(INDEX_BIN_ARRAY(bin_sums, m_index));)
      }

      __syncthreads();

      
      while (num_to_reduce > 1) {
        /* uint32_t bit0 = count.bit0; */
      }
#undef INDEX_BIN_ARRAY
#undef BIN_LOOP
    }
  }
}

int main() {
  int ngpus = 0;
  cudaGetDeviceCount(&ngpus);
  printf("ngpus = %d\n", ngpus);
  if (ngpus > 0)
    cudaSetDevice(0);
  else
    return 0;

  uint32_t m = 10;
  uint32_t n = 100;

  std::vector<uint32_t> range_to_m;

  for (uint32_t i = 0; i < m; ++i) {
    range_to_m.push_back(i);
  }

  uint32_t **data;
  uint32_t **max_vals;
  uint32_t **max_locs;

  /* #ifdef USE_NVTX */
  /*   // nvtxRangePushA("A"); */
  /*   nvtxRangeId_t nvtx_1 = nvtxRangeStartA("A"); */
  /* #endif */

  cudaMallocManaged(&data, m * sizeof(uint32_t *));
  cudaMallocManaged(&max_vals, m * sizeof(uint32_t *));
  cudaMallocManaged(&max_locs, m * sizeof(size_t *));

  cudaMallocManaged(&data[0], n * m * sizeof(uint32_t));
  cudaMallocManaged(&max_vals[0], n * m * sizeof(uint32_t));
  cudaMallocManaged(&max_locs[0], n * m * sizeof(size_t));

  for (size_t i = 1; i < m; ++i) {
    data[i] = data[0] + i * n;
    max_vals[i] = max_vals[0] + i * n;
    max_locs[i] = max_locs[0] + i * n;
  }

  /* #ifdef USE_NVTX */
  /*   nvtxRangeEnd(nvtx_1); */
  /*   //nvtxRangePop(); */
  /* #endif */

  for (unsigned i = 0; i < n; ++i) {
    std::random_shuffle(range_to_m.begin(), range_to_m.end());
    for (unsigned j = 0; j < m; ++j) {
      data[i][j] = range_to_m[j];
    }
  }

  uint32_t nth_max = static_cast<uint32_t>(std::rand()) % m;

  /* ---------------  TASK 1  ------------ */

  std::cout << "==== cpu ====\n";
  for (size_t i = 0; i < n; i++) {
    std::cout << "value: " << max_vals[i][0] << " loc: " << max_locs[i][0]
              << " value nth: " << max_vals[i][1]
              << " loc nth: " << max_locs[i][1] << "\n";
  }
  std::cout << std::endl;

  /* ---------------  TASK 2  ------------ */

  /* ---------------  TASK 3  ------------ */

  // write GPU code to find the maximum in each row of data, i.e  MAX(data[i])
  // for each i also find the locaiton of each maximum

  // write GPU code to find the first maximum and the Nth maximum value in each
  // row of data, i.e  MAX(data[i]) for each i also find the locaiton of each
  // maximum

  maxes<<<1, 1>>>(data, max_vals, max_locs, n, m);

  std::cout << "==== gpu ====\n";
  for (size_t i = 0; i < n; i++) {
    std::cout << "value: " << max_vals[i][0] << " loc: " << max_locs[i][0]
              << " value nth: " << max_vals[i][1]
              << " loc nth: " << max_locs[i][1] << "\n";
  }
  std::cout << std::endl;

  // print results;
  cudaDeviceSynchronize();

  cudaFree(data[0]);
  cudaFree(data);

  return 0;
}
