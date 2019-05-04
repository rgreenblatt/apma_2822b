#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <vector>

#define cuda_error_chk(ans)                                                    \
  { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

#define FULL_MASK 0xffffffff

// primarily for camel case avoidance
const unsigned WARP_SIZE = 32;
const unsigned LOG_WARP_SIZE = 5;

template <typename T>
__forceinline__ __device__ T warp_prefix_sum(T val, T *sum,
                                             unsigned thread_idx) {
  T orig_val = val;

  for (uint8_t i = 1; i < WARP_SIZE; i <<= 1) {
    auto adder = __shfl_up_sync(FULL_MASK, val, i);

    if (thread_idx % WARP_SIZE >= i) {
      val += adder;
    }
  }

  if (sum != NULL) {
    *sum = val;
  }

  return val - orig_val;
}

union loc_value {
  struct {
    uint32_t val;
    unsigned idx;
  };
  uint64_t bytes;
};

__forceinline__ __device__ loc_value warp_reduce_max(loc_value p) {
  for (unsigned offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    loc_value other_p;
    other_p.bytes = __shfl_down_sync(FULL_MASK, p.bytes, offset);

    if (other_p.val > p.val) {
      p = other_p;
    }
  }

  return p;
}

inline unsigned partition(loc_value *vals, unsigned start, unsigned end,
                   unsigned index) {
  auto pivot = vals[index];
  vals[index] = vals[end - 1];
  vals[end - 1] = pivot;
  unsigned idx_lower = start;
  for (unsigned i = start; i < end - 1; ++i) {
    if (vals[i].val < pivot.val) {
      auto tmp = vals[idx_lower];
      vals[idx_lower] = vals[i];
      vals[i] = tmp;
      idx_lower++;
    }
  }
  auto tmp = vals[idx_lower];
  vals[idx_lower] = vals[end - 1];
  vals[end - 1] = tmp;
  return idx_lower;
}

inline loc_value quick_select(loc_value *vals, unsigned start,
                                 unsigned end, unsigned nth) {
  if (start - end == 1) {
    return vals[start];
  }

  unsigned pivot_index = start;
  pivot_index = partition(vals, start, end, pivot_index);

  if (pivot_index == nth) {
    return vals[pivot_index];
  } else if (pivot_index > nth) {
    return quick_select(vals, start, pivot_index, nth);
  } else {
    return quick_select(vals, pivot_index + 1, end, nth);
  }
}

extern __shared__ uint32_t shared_memory[];

__global__ void max_data(loc_value **data, uint32_t **max_vals, uint32_t **max_locs,
                    unsigned m) {

  loc_value *maxes = (loc_value *)shared_memory;

  size_t n_idx = blockIdx.x;
  unsigned m_index = threadIdx.x;
  unsigned warp_idx = threadIdx.x >> LOG_WARP_SIZE;

  bool first_iter = true;
  unsigned size_reduced = m;
  loc_value warp_max;

  do {
    if (first_iter) {
      __syncthreads();
    }

    unsigned next_size_reduced = (size_reduced - 1 + WARP_SIZE) / WARP_SIZE;

    if (m_index < size_reduced) {

      warp_max = warp_reduce_max(first_iter ? data[n_idx][m_index]
                                            : maxes[m_index]);

      if (next_size_reduced == 1) {
        if (!m_index) {
          max_vals[n_idx][0] = warp_max.val;
          max_locs[n_idx][0] = warp_max.idx;
          printf("idx: %u, n: %lu, val is: %u\n", m_index, n_idx,
                 max_vals[n_idx][0]);
        }
      } else if (m_index % WARP_SIZE) {
        maxes[warp_idx] = warp_max;
      }

      first_iter = false;
    }

    size_reduced = next_size_reduced;

  } while (size_reduced > 1);
  /* printf("idx: %u, n: %lu, val is: %u\n", m_index, n_idx, */
  /*        max_vals[n_idx][0]); */
}

const uint8_t BITS_PER_PASS = 2;
const uint8_t BINS_PER_PASS = 4;

union packed {
  uint32_t bytes;
  uint8_t vals[BINS_PER_PASS];
};

__global__ void sort_maxes(uint32_t **data, uint32_t **max_vals, uint32_t **max_locs,
                      uint32_t m, uint32_t nth, uint32_t num_warps_per_n) {

  uint32_t *bin_sums = shared_memory;

  size_t current_size = num_warps_per_n * BINS_PER_PASS;
  uint32_t *working_data = &bin_sums[current_size];

  current_size += m;
  uint32_t *sorted_data = &bin_sums[current_size];

  current_size += m;
  uint32_t *indexes = &bin_sums[current_size];

  current_size += m;
  uint32_t *sorted_indexes = &bin_sums[current_size];

  current_size += m;
  uint32_t *n_maxes = &bin_sums[current_size];

  current_size += BINS_PER_PASS;
  packed *warp_maxes = (packed *)&bin_sums[current_size];

  current_size += num_warps_per_n;

  uint32_t m_per_warp = (m - 1 + num_warps_per_n) / num_warps_per_n;
  uint32_t m_iterations = (m_per_warp - 1 + WARP_SIZE) / WARP_SIZE;

  packed *warp_iteration_maxes = (packed *)&bin_sums[current_size];
  current_size += num_warps_per_n * m_iterations;

  size_t n_idx = blockIdx.x;
  uint32_t m_index = threadIdx.x;
  unsigned warp_idx = threadIdx.x >> LOG_WARP_SIZE;

  uint32_t m_local_index = m_index % WARP_SIZE;
  uint32_t m_offset = m_per_warp * warp_idx;
  uint32_t m_start = m_local_index + m_offset;
  uint32_t m_end = umin(m, m_per_warp * (warp_idx + 1));

  for (uint32_t j = m_start; j < m_end; j += WARP_SIZE) {
    working_data[j] = data[n_idx][j];
    indexes[j] = j;
  }

  bool is_last_thread_in_warp = (m_index + 1) % WARP_SIZE == 0;

  packed *counts = new packed[m_iterations]();

  for (uint8_t shift = 0; shift < 8 * sizeof(uint32_t);
       shift += BITS_PER_PASS) {

    for (uint32_t i = 0; i < m_iterations; ++i) {
      counts[i].bytes = 0;
    }

    for (size_t j = m_start; j < m_end; j += WARP_SIZE) {
      counts[(j - m_offset) / WARP_SIZE]
          .vals[(working_data[j] >> shift) & (BINS_PER_PASS - 1)]++;
      counts[(j - m_offset) / WARP_SIZE].bytes = warp_prefix_sum(
          counts[(j - m_offset) / WARP_SIZE].bytes,
          (is_last_thread_in_warp || j == m_end - 1)
              ? &warp_iteration_maxes[warp_idx * m_iterations +
                                      (j - m_offset) / WARP_SIZE]
                     .bytes
              : nullptr,
          threadIdx.x);
    }

    if (m_local_index < m_iterations) {
      warp_iteration_maxes[warp_idx * m_iterations + m_local_index]
          .bytes = warp_prefix_sum(
          warp_iteration_maxes[warp_idx * m_iterations + m_local_index].bytes,
          (m_local_index == m_iterations - 1) ? &warp_maxes[warp_idx].bytes
                                              : nullptr,
          threadIdx.x);
    }

    if (is_last_thread_in_warp) {
      for (uint32_t bin = 0; bin < BINS_PER_PASS; bin++) {
        bin_sums[BINS_PER_PASS * warp_idx + bin] =
            warp_maxes[warp_idx].vals[bin];
        /* printf("warp_idx: %u, warp max: %u\n", warp_idx, */
        /*        warp_maxes[warp_idx].array_int[bin]); */
      }
    }

    __syncthreads();

    // assumption is that num_warps_per_n is less than the WARP_SIZE
    if (m_index < num_warps_per_n) {
      for (uint8_t bin = 0; bin < BINS_PER_PASS; bin++) {
        bin_sums[BINS_PER_PASS * m_index + bin] = warp_prefix_sum(
            bin_sums[BINS_PER_PASS * m_index + bin],
            (m_index == num_warps_per_n - 1) ? &n_maxes[bin] : (uint32_t *)0,
            threadIdx.x);
      }
    }

    if (m_index < BINS_PER_PASS) {
      n_maxes[m_index] =
          warp_prefix_sum(n_maxes[m_index], (uint32_t *)0, threadIdx.x);
    }

    __syncthreads();

    for (size_t j = m_start; j < m_end; j += WARP_SIZE) {
      uint8_t bin = (working_data[j] >> shift) & (BINS_PER_PASS - 1);
      uint32_t idx = counts[(j - m_offset) / WARP_SIZE].vals[bin] +
                     bin_sums[BINS_PER_PASS * warp_idx + bin] + n_maxes[bin] +
                     warp_iteration_maxes[warp_idx * m_iterations +
                                          (j - m_offset) / WARP_SIZE]
                         .vals[bin];

      sorted_data[idx] = working_data[j];
      sorted_indexes[idx] = indexes[j];
    }

    __syncthreads();

    uint32_t *temp_ptr = working_data;
    working_data = sorted_data;
    sorted_data = temp_ptr;

    temp_ptr = indexes;
    indexes = sorted_indexes;
    sorted_indexes = temp_ptr;
  }

  if (m_end == m && m_start < m) {
    max_locs[n_idx][0] = indexes[m - 1];
    max_vals[n_idx][0] = sorted_data[m - 1];
  }
  if (nth >= m_start && nth < m_end) {
    max_locs[n_idx][1] = indexes[nth];
    max_vals[n_idx][1] = sorted_data[nth];
  }

  delete[] counts;
}

template <typename T> void fill(T *arr, size_t size, T val) {
  for (size_t i = 0; i < size; ++i) {
    arr[i] = val;
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

  uint32_t m = 64;
  /* uint32_t n = 16384; */
  uint32_t n = 1;

  std::vector<uint32_t> range_to_m;

  for (uint32_t i = 0; i < m; ++i) {
    range_to_m.push_back(i);
  }

  uint32_t **data;
  loc_value **data_struct;
  uint32_t **max_vals;
  uint32_t **max_locs;
  uint32_t **cpu_max_locs;

  cudaMallocManaged(&data, n * sizeof(uint32_t *));
  cudaMallocManaged(&data_struct, n * sizeof(loc_value *));
  cudaMallocManaged(&max_vals, n * sizeof(uint32_t *));
  cudaMallocManaged(&max_locs, n * sizeof(size_t *));
  cpu_max_locs = new uint32_t *[n];

  cudaMallocManaged(&data[0], n * m * sizeof(uint32_t));
  cudaMallocManaged(&data_struct[0], n * m * sizeof(loc_value));
  cudaMallocManaged(&max_vals[0], n * m * sizeof(uint32_t));
  cudaMallocManaged(&max_locs[0], n * m * sizeof(size_t));
  cpu_max_locs[0] = new uint32_t[n * m];

  for (size_t i = 1; i < n; ++i) {
    data[i] = data[0] + i * m;
    data_struct[i] = data_struct[0] + i * m;
    max_vals[i] = max_vals[0] + i * m;
    max_locs[i] = max_locs[0] + i * m;
    cpu_max_locs[i] = cpu_max_locs[0] + i * m;
  }

  // for deterministic shuffles, comment out the below lines
  std::random_device r;
  std::srand(r());

  for (unsigned i = 0; i < n; ++i) {
    std::random_shuffle(range_to_m.begin(), range_to_m.end());
    for (unsigned j = 0; j < m; ++j) {
      data[i][j] = range_to_m[j];
      data_struct[i][j].val = range_to_m[j];
      data_struct[i][j].idx = j;
    }
  }

  uint32_t nth_max = static_cast<uint32_t>(std::rand()) % m;

  /* ---------------  TASK 1  ------------ */

  {
    fill(cpu_max_locs[0], n * 2, static_cast<uint32_t>(-1));
    fill(max_vals[0], n * 2, 0u);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      for (uint32_t j = 0; j < m; ++j) {
        if (data[i][j] > max_vals[i][0]) {
          max_vals[i][0] = data[i][j];
          cpu_max_locs[i][0] = j;
        }
      }
    }

    auto **vals = new loc_value*[n];
    vals[0] = new loc_value[n * m];
    for (unsigned i = 1; i < n; ++i) {
      vals[i] = vals[0] + i * m;
    }

    for (uint32_t i = 0; i < n; ++i) {
      for (uint32_t j = 0; j < m; ++j) {
        vals[i][j].idx = j;
        vals[i][j].val = data[i][j];
      }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      auto selected = quick_select(vals[i], 0, m, nth_max);
      max_vals[i][1] = selected.val;
      cpu_max_locs[i][1] = selected.idx;
    }

    for (size_t i = 0; i < n; ++i) {
      assert(max_vals[i][0] == m - 1);
      assert(max_vals[i][1] == nth_max);
    }

    std::cout << "==== cpu passed tests ====" << std::endl;
  }

  /* ---------------  TASK 2  ------------ */

  { 
    fill(max_locs[0], n * 2, static_cast<uint32_t>(-1));
    fill(max_vals[0], n * 2, static_cast<uint32_t>(-1));

    max_data<<<n, m, m * sizeof(loc_value)>>>(data_struct, max_locs, max_vals,
                                              m);

    cuda_error_chk(cudaDeviceSynchronize());

    printf("val is: %u\n", max_vals[0][0]);

    for (size_t i = 0; i < n; ++i) {
      /* printf("val is: %u\n", max_vals[0][0]); */
      /* printf("val is: %u\n", max_vals[i][0]); */
      assert(max_vals[i][0] == m - 1);
      /* printf("val is: %u\n", max_vals[0][0]); */
      /* printf("val is: %u\n", max_vals[i][0]); */
      assert(max_locs[i][0] == cpu_max_locs[i][0]);
    }

    
  }

  /* ---------------  TASK 3  ------------ */

  {
    fill(max_locs[0], n * 2, static_cast<uint32_t>(-1));
    fill(max_vals[0], n * 2, static_cast<uint32_t>(-1));

    uint32_t num_threads_per_block = 256;
    uint32_t num_warps_per_n =
        std::min((num_threads_per_block + WARP_SIZE - 1) / WARP_SIZE,
                 ((m + WARP_SIZE - 1) / WARP_SIZE));

    uint32_t m_per_warp = m / num_warps_per_n;
    uint32_t m_iterations = (m_per_warp - 1 + WARP_SIZE) / WARP_SIZE;

    // assumptions:
    // - values per warp < 256
    // - num_warps_per_n is less than the WARP_SIZE
    // - num theads is a multiple of the warp size

    assert(m_per_warp < static_cast<uint8_t>(-1));
    assert(num_threads_per_block / WARP_SIZE < WARP_SIZE);
    assert(num_threads_per_block % WARP_SIZE == 0);

    size_t shared_count = num_warps_per_n * BINS_PER_PASS + num_warps_per_n +
                          +num_warps_per_n * m_iterations + m * 4 +
                          BINS_PER_PASS;

    sort_maxes<<<n, num_threads_per_block, shared_count * sizeof(int32_t)>>>(
        data, max_vals, max_locs, m, nth_max, num_warps_per_n);

    cuda_error_chk(cudaDeviceSynchronize());

    for (size_t i = 0; i < n; ++i) {
      assert(max_vals[i][0] == m - 1);
      assert(max_vals[i][1] == nth_max);
      assert(max_locs[i][0] == cpu_max_locs[i][0]);
      assert(max_locs[i][1] == cpu_max_locs[i][1]);
    }

    std::cout << "==== gpu passed tests ====" << std::endl;
  }

  // print results;
  cudaDeviceSynchronize();

  cudaFree(data[0]);
  cudaFree(data);

  return 0;
}
