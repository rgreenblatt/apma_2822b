#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <utility>
#include <vector>
#include <stdlib.h>

#define cuda_error_chk(ans)                                                    \
  { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

#define USE_NVTX

#ifdef USE_NVTX
#include "nvToolsExt.h"
#endif

#define FULL_MASK 0xffffffff

// primarily for camel case avoidance
enum : uint32_t { warp_size = 32, log_warp_size = 5 };

template <typename T>
__forceinline__ __device__ T warp_prefix_sum(T val, T *sum,
                                             uint32_t thread_idx) {
  T orig_val = val;

  for (uint8_t i = 1; i < warp_size; i <<= 1) {
    auto adder = __shfl_up_sync(FULL_MASK, val, i);

    if (thread_idx % warp_size >= i) {
      val += adder;
    }
  }

  if (sum != NULL) {
    *sum = val;
  }

  return val - orig_val;
}

__global__ void test_prefix_sum(uint32_t *data, uint32_t *returned,
                                uint32_t n) {
  uint32_t sum;
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    returned[index] = warp_prefix_sum(data[index], &sum, threadIdx.x);
    printf("adding index: %lu, as: %u, sum: %u\n", index, returned[index], sum);
  }
}

#define BITS_PER_PASS 2
#define BINS_PER_PASS 4

union packed
{
  uint32_t packed_int;
  uint8_t array_int[BINS_PER_PASS];
};

__global__ void maxes(uint32_t **data, uint32_t **max_vals, uint32_t **max_locs,
                      uint32_t n, uint32_t m, u_int32_t nth,
                      uint32_t total_count,
                      uint32_t num_warps_per_n) {

  extern __shared__ uint32_t shared_memory[];

  uint32_t *bin_sums = shared_memory;
  size_t size_bins = n * num_warps_per_n * BINS_PER_PASS;

  size_t current_size = size_bins;
  uint32_t *working_data = &bin_sums[current_size];

  size_t size_data = n * m;
  current_size += size_data;
  uint32_t *sorted_data = &bin_sums[current_size];

  current_size += size_data;
  uint32_t *indexes = &bin_sums[current_size];

  current_size += size_data;
  uint32_t *sorted_indexes = &bin_sums[current_size];

  current_size += size_data;
  uint32_t *n_maxes = &bin_sums[current_size];

  current_size += BINS_PER_PASS * n;
  packed *warp_maxes = (packed *)&bin_sums[current_size];

  size_t size_bins_warp = n * num_warps_per_n;
  current_size += size_bins_warp;

  uint32_t m_per_warp = m / num_warps_per_n;  
  uint32_t m_iterations = (m_per_warp - 1 + warp_size) / warp_size;

  packed *warp_iteration_maxes = (packed *)&bin_sums[current_size];
  current_size += size_bins_warp * m_iterations;

  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  size_t n_start = (index * n) / total_count;
  size_t n_stop_iter = ullmin(((index + 1) * n) / total_count + 1, n);

  uint32_t m_index = static_cast<uint32_t>(index - (n_start * total_count) / n);

  auto warp_id = threadIdx.x >> log_warp_size;

  uint32_t m_local_index = m_index % warp_size;

  uint32_t m_offset = m_per_warp * warp_id;
  uint32_t m_start = m_local_index + m_offset;
  uint32_t m_end =
      (warp_id == num_warps_per_n - 1) ? m : m_per_warp * (warp_id + 1);

  for (size_t i = n_start; i < n_stop_iter; ++i) {
    for (uint32_t j = m_start; j < m_end; j += warp_size) {
      indexes[i * m + j] = j;
      working_data[i * m + j] = data[i][j];
    }
  }

  __syncthreads();

  bool is_last_thread_in_warp = (m_index + 1) % warp_size == 0;

  for (uint8_t shift = 0; shift < 8 * sizeof(uint32_t); shift += BITS_PER_PASS) {
    for (size_t i = n_start; i < n_stop_iter; ++i) {

#define INDEX_BIN_ARRAY(arr, iter_index)                                       \
  (arr[i * num_warps_per_n * BINS_PER_PASS + iter_index * BINS_PER_PASS + bin])

      auto warp_idx = (threadIdx.x >> log_warp_size) + i * num_warps_per_n;

      packed *counts = new packed[m_iterations]();

      for (size_t j = m_start; j < m_end; j += warp_size) {
        counts[(j - m_offset) / warp_size].array_int[(working_data[i * m + j] >> shift) &
                                        (BINS_PER_PASS - 1)]++;
        counts[(j - m_offset) / warp_size].packed_int = warp_prefix_sum(
            counts[(j - m_offset) / warp_size].packed_int,
            (is_last_thread_in_warp || j == m_end - 1)
                ? &warp_iteration_maxes[warp_idx * m_iterations + (j - m_offset) / warp_size]
                       .packed_int
                : nullptr,
            threadIdx.x);
      }

      if (m_local_index < m_iterations) {
        warp_iteration_maxes[warp_idx * m_iterations + m_local_index]
            .packed_int = warp_prefix_sum(
            warp_iteration_maxes[warp_idx * m_iterations + m_local_index]
                .packed_int,
            (m_local_index == m_iterations - 1)
                ? &warp_maxes[warp_idx].packed_int
                : nullptr,
            threadIdx.x);
      }

      if (is_last_thread_in_warp) {
        for (uint32_t bin = 0; bin < BINS_PER_PASS; bin++) {
          INDEX_BIN_ARRAY(bin_sums, warp_id) =
              warp_maxes[warp_idx].array_int[bin];
        }
      }

    __syncthreads();

    // assumption is that num_warps_per_n is less than the warp_size
    if (m_index < num_warps_per_n) {
      for (uint8_t bin = 0; bin < BINS_PER_PASS; bin++) {
        INDEX_BIN_ARRAY(bin_sums, m_index) = warp_prefix_sum(
            INDEX_BIN_ARRAY(bin_sums, m_index),
            (m_index == num_warps_per_n - 1) ? &n_maxes[i * BINS_PER_PASS + bin]
                                             : (uint32_t *)0,
            threadIdx.x);
      }
    }

   if (m_index < BINS_PER_PASS) {
      n_maxes[i * BINS_PER_PASS + m_index] = warp_prefix_sum(
          n_maxes[i * BINS_PER_PASS + m_index], (uint32_t *)0, threadIdx.x);
    }
    __syncthreads();

    for (size_t j = m_start; j < m_end; j += warp_size) {
      uint8_t bin = (working_data[i * m + j] >> shift) & (BINS_PER_PASS - 1);
      uint32_t idx =
          counts[(j - m_offset) / warp_size].array_int[bin] +
          INDEX_BIN_ARRAY(bin_sums, warp_id) +
          n_maxes[i * BINS_PER_PASS + bin] +
          warp_iteration_maxes[warp_idx * m_iterations + (j - m_offset) / warp_size]
              .array_int[bin];

      sorted_data[i * m + idx] = working_data[i * m + j];
      sorted_indexes[i * m + idx] = indexes[i * m + j];
    }

    __syncthreads();

    uint32_t *temp_ptr = working_data;
    working_data = sorted_data;
    sorted_data = temp_ptr;

    temp_ptr = indexes;
    indexes = sorted_indexes;
    sorted_indexes = temp_ptr;



#undef INDEX_BIN_ARRAY
#undef BIN_LOOP
      }
  }


  for (size_t i = n_start; i < n_stop_iter; ++i) {
    if (m_start == 0) {
      max_locs[n_start][0] = indexes[i * m];
      max_vals[n_start][0] = sorted_data[i * m];
    }
    if (nth >= m_start && nth < m_end) {
      max_locs[n_start][1] = indexes[nth + i * m];
      max_vals[n_start][1] = sorted_data[nth + i * m];
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

  uint32_t m = 1024;
  uint32_t n = 1;

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

  cudaMallocManaged(&data, n * sizeof(uint32_t *));
  cudaMallocManaged(&max_vals, n * sizeof(uint32_t *));
  cudaMallocManaged(&max_locs, n * sizeof(size_t *));

  cudaMallocManaged(&data[0], n * m * sizeof(uint32_t));
  cudaMallocManaged(&max_vals[0], n * m * sizeof(uint32_t));
  cudaMallocManaged(&max_locs[0], n * m * sizeof(size_t));

  for (size_t i = 1; i < n; ++i) {
    data[i] = data[0] + i * m;
    max_vals[i] = max_vals[0] + i * m;
    max_locs[i] = max_locs[0] + i * m;
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

  // set to obvious failure
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      max_locs[i][j] = static_cast<uint32_t>(-1);
      max_vals[i][j] = static_cast<uint32_t>(-1);
    }
  }

  uint32_t nth_max = static_cast<uint32_t>(std::rand()) % m;
  std::cout << "nth is " << nth_max << std::endl;

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

  // set to obvious failure
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      max_locs[i][j] = static_cast<uint32_t>(-1);
      max_vals[i][j] = static_cast<uint32_t>(-1);
    }
  }

  //test code
  /* const uint test_size = 16; */
  /* uint32_t *data_test; */
  /* uint32_t *return_test; */
  /* cudaMallocManaged(&data_test, n * sizeof(uint32_t)); */
  /* cudaMallocManaged(&return_test, n * sizeof(uint32_t)); */

  /* for (size_t i = 0; i < test_size; ++i) { */
  /*   return_test[i] = static_cast<uint32_t>(-1); */
  /*   data_test[i] = i; */
  /* } */

  /* test_prefix_sum<<<1, test_size>>>(data_test, return_test, test_size); */

  /* cuda_error_chk(cudaDeviceSynchronize()); */

  /* for (size_t i = 0; i < test_size; ++i) { */
  /*   printf("i: %lu, r: %u\n", i, return_test[i]); */
  /* } */

  //assumptions:
  // - values per warp < 256
  // - num_warps_per_n is less than the warp_size

  uint32_t num_threads_per_block = 128;
  uint32_t num_warps_per_n =
      std::min((num_threads_per_block + warp_size - 1) / warp_size,
               ((m + warp_size - 1) / warp_size));

  size_t shared_count = (n * num_warps_per_n * BINS_PER_PASS) / 4 +
                        n * num_warps_per_n * BINS_PER_PASS + n * m * 4 +
                        BINS_PER_PASS * n;

  maxes<<<n, num_threads_per_block, shared_count * sizeof(int32_t)>>>(
      data, max_vals, max_locs, n, m, nth_max, n * num_threads_per_block,
      num_warps_per_n);

  cuda_error_chk(cudaDeviceSynchronize());

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
