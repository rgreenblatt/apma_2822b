#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
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

#define CASSERT(predicate, file) _impl_CASSERT_LINE(predicate,__LINE__,file)

#define _impl_PASTE(a,b) a##b
#define _impl_CASSERT_LINE(predicate, line, file) \
    typedef char _impl_PASTE(assertion_failed_##file##_,line)[2*!!(predicate)-1];

namespace chr = std::chrono;
using h_clock = chr::high_resolution_clock;

#define FULL_MASK 0xffffffff

// primarily for camel case avoidance
const unsigned WARP_SIZE = 32;
const unsigned LOG_WARP_SIZE = 5;

template <typename T>
__forceinline__ __device__ T warp_prefix_sum(T val, T *sum, unsigned thread_idx,
                                             unsigned mask) {
  T orig_val = val;

  for (unsigned i = 1; i < WARP_SIZE; i <<= 1) {
    auto adder = __shfl_up_sync(mask, val, i);

    if (thread_idx % WARP_SIZE >= i) {
      val += adder;
    }
  }

  if (sum != NULL) {
    *sum = val;
  }

  return val - orig_val;
}

using d_type = uint32_t;

CASSERT(sizeof(d_type) == 4, d_type_size);

union loc_value {
  struct {
    d_type val;
    unsigned idx;
  };
  uint64_t bytes;
};

__forceinline__ __device__ loc_value warp_reduce_max(loc_value p,
                                                     unsigned mask) {
  for (unsigned offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    loc_value other_p;
    other_p.bytes = __shfl_down_sync(mask, p.bytes, offset);

    if (other_p.val > p.val) {
      p = other_p;
    }
  }

  return p;
}

inline void swap(loc_value *data, unsigned idx1, unsigned idx2,
                 loc_value &val) {
  val = data[idx1];
  data[idx1] = data[idx2];
  data[idx2] = val;
}

inline void swap(loc_value *data, unsigned idx1, unsigned idx2) {
  loc_value val;
  swap(data, idx1, idx2, val);
}

inline unsigned partition(loc_value *data, unsigned start, unsigned end,
                          unsigned index) {
  loc_value pivot;
  swap(data, index, end - 1, pivot); 
  unsigned idx_lower = start;
  for (unsigned i = start; i < end - 1; ++i) {
    if (data[i].val < pivot.val) {
      swap(data, idx_lower, i);
      idx_lower++;
    }
  }
  swap(data, idx_lower, end - 1);
  return idx_lower;
}

inline loc_value quick_select(loc_value *data, unsigned start, unsigned end,
                              unsigned nth) {
  if (start - end == 1) {
    return data[start];
  }

  unsigned pivot_index = start;
  pivot_index = partition(data, start, end, pivot_index);

  if (pivot_index == nth) {
    return data[pivot_index];
  } else if (pivot_index > nth) {
    return quick_select(data, start, pivot_index, nth);
  } else {
    return quick_select(data, pivot_index + 1, end, nth);
  }
}

extern __shared__ d_type shared_memory[];

__global__ void get_max_warp_per_n(loc_value **data, d_type **max_vals,
                                unsigned **max_locs, unsigned m) {

  unsigned n_idx = blockIdx.x;
  unsigned m_index = threadIdx.x;

  loc_value max_val; 

  for (unsigned i = m_index; i < m; i+=WARP_SIZE) {
    if (i == m_index) {
      max_val = data[n_idx][i];
    } else if (data[n_idx][i].val > max_val.val) {
      max_val = data[n_idx][i];
    }
  }

  loc_value overall_max = warp_reduce_max(max_val, FULL_MASK);

  if (!m_index) {
    max_vals[n_idx][0] = overall_max.val;
    max_locs[n_idx][0] = overall_max.idx;
  }
}
__global__ void get_max(loc_value **data, d_type **max_vals,
                         unsigned **max_locs, unsigned m) {

  loc_value *maxes = (loc_value *)shared_memory;

  unsigned n_idx = blockIdx.x;
  unsigned m_index = threadIdx.x;
  unsigned warp_idx = threadIdx.x >> LOG_WARP_SIZE;

  bool first_iter = true;
  unsigned size_reduced = m;
  loc_value warp_max;


  do {
    if (!first_iter) {
      __syncthreads();
    }

    unsigned next_size_reduced = (size_reduced - 1 + WARP_SIZE) / WARP_SIZE;

    unsigned mask = __ballot_sync(FULL_MASK, m_index < size_reduced);

    if (m_index < size_reduced) {

      warp_max = warp_reduce_max(
          first_iter ? data[n_idx][m_index] : maxes[m_index], mask);

      if (next_size_reduced == 1) {
        if (!m_index) {
          max_vals[n_idx][0] = warp_max.val;
          max_locs[n_idx][0] = warp_max.idx;
        }
      } else if (!(m_index % WARP_SIZE)) {
        maxes[warp_idx] = warp_max;
      }
    }

    first_iter = false;

    size_reduced = next_size_reduced;

  } while (size_reduced > 1);
}

const uint8_t BITS_PER_PASS = 2;

//must be 4 for packing to work
const uint8_t BINS_PER_PASS = 4;

union packed {
  uint32_t bytes;
  uint8_t vals[BINS_PER_PASS];
};

__forceinline__ __device__ unsigned get_bin(d_type val, unsigned shift) {
  return (val >> shift) & (BINS_PER_PASS - 1);
}

__global__ void sort_maxes(d_type **data, d_type **max_vals,
                           unsigned **max_locs, unsigned m, unsigned nth,
                           unsigned num_warps_per_n) {

  d_type *sum_over_warps = shared_memory;

  unsigned current_size = num_warps_per_n * BINS_PER_PASS;
  d_type *working_data = &sum_over_warps[current_size];

  current_size += m;
  d_type *sorted_data = &sum_over_warps[current_size];

  current_size += m;
  unsigned *indexes = (unsigned *)&sum_over_warps[current_size];

  current_size += m;
  unsigned *sorted_indexes = (unsigned *)&sum_over_warps[current_size];

  current_size += m;
  d_type *sum_over_n = &sum_over_warps[current_size];

  current_size += BINS_PER_PASS;
  packed *sum_over_warps_packed = (packed *)&sum_over_warps[current_size];

  current_size += num_warps_per_n;

  unsigned m_per_warp = (m - 1 + num_warps_per_n) / num_warps_per_n;
  unsigned m_iterations = (m_per_warp - 1 + WARP_SIZE) / WARP_SIZE;

  packed *sum_within_warp = (packed *)&sum_over_warps[current_size];
  current_size += num_warps_per_n * m_iterations;

  unsigned n_idx = blockIdx.x;
  unsigned m_index = threadIdx.x;
  unsigned warp_idx = threadIdx.x >> LOG_WARP_SIZE;

  unsigned m_local_index = m_index % WARP_SIZE;
  unsigned m_offset = m_per_warp * warp_idx;
  unsigned m_start = m_local_index + m_offset;
  unsigned m_end = umin(m, m_per_warp * (warp_idx + 1));

  for (unsigned j = m_start; j < m_end; j += WARP_SIZE) {
    working_data[j] = data[n_idx][j];
    indexes[j] = j;
  }

  bool is_last_thread_in_warp = (m_index + 1) % WARP_SIZE == 0;

  packed *counts = new packed[m_iterations]();

  for (unsigned shift = 0; shift < 8 * sizeof(d_type); shift += BITS_PER_PASS) {

    for (unsigned i = 0; i < m_iterations; ++i) {
      counts[i].bytes = 0;
    }

    unsigned mask_warp_reduce = __ballot_sync(FULL_MASK, m_start < m_end);

    for (unsigned j = m_start; j < m_end; j += WARP_SIZE) {
      unsigned iteration = (j - m_offset) / WARP_SIZE;

      counts[iteration].vals[get_bin(working_data[j], shift)]++;

      counts[iteration].bytes = warp_prefix_sum(
          counts[iteration].bytes,
          (is_last_thread_in_warp || j == m_end - 1)
              ? &sum_within_warp[warp_idx * m_iterations + iteration].bytes
              : nullptr,
          threadIdx.x, mask_warp_reduce);

      mask_warp_reduce = __ballot_sync(FULL_MASK, j + WARP_SIZE < m_end);
    }

    bool iteration_reduce_condition = m_local_index < m_iterations;

    unsigned mask_warp_iteration_reduce =
        __ballot_sync(FULL_MASK, iteration_reduce_condition);

    if (iteration_reduce_condition) {
      sum_within_warp[warp_idx * m_iterations + m_local_index]
          .bytes = warp_prefix_sum(
          sum_within_warp[warp_idx * m_iterations + m_local_index].bytes,
          (m_local_index == m_iterations - 1) ? &sum_over_warps_packed[warp_idx].bytes
                                              : nullptr,
          threadIdx.x, mask_warp_iteration_reduce);
    }

    if (is_last_thread_in_warp) {
      for (unsigned bin = 0; bin < BINS_PER_PASS; bin++) {
        sum_over_warps[BINS_PER_PASS * warp_idx + bin] =
            sum_over_warps_packed[warp_idx].vals[bin];
      }
    }

    __syncthreads();

    bool bin_reduce_condition = m_index < num_warps_per_n;

    unsigned mask_warp_bin_reduce =
        __ballot_sync(FULL_MASK, bin_reduce_condition);

    // assumption is that num_warps_per_n is less than the WARP_SIZE
    if (bin_reduce_condition) {
      for (unsigned bin = 0; bin < BINS_PER_PASS; bin++) {
        sum_over_warps[BINS_PER_PASS * m_index + bin] = warp_prefix_sum(
            sum_over_warps[BINS_PER_PASS * m_index + bin],
            (m_index == num_warps_per_n - 1) ? &sum_over_n[bin] : (d_type *)0,
            threadIdx.x, mask_warp_bin_reduce);
      }
    }

    bool n_reduce_condition = m_index < BINS_PER_PASS;

    unsigned mask_warp_n_reduce = __ballot_sync(FULL_MASK, n_reduce_condition);

    if (n_reduce_condition) {
      sum_over_n[m_index] = warp_prefix_sum(sum_over_n[m_index], (d_type *)0,
                                         threadIdx.x, mask_warp_n_reduce);
    }

    __syncthreads();

    for (unsigned j = m_start; j < m_end; j += WARP_SIZE) {
      unsigned iteration = (j - m_offset) / WARP_SIZE;

      unsigned bin = get_bin(working_data[j], shift);
      unsigned idx =
          counts[iteration].vals[bin] +
          sum_over_warps[BINS_PER_PASS * warp_idx + bin] + sum_over_n[bin] +
          sum_within_warp[warp_idx * m_iterations + iteration].vals[bin];

      sorted_data[idx] = working_data[j];
      sorted_indexes[idx] = indexes[j];
    }

    __syncthreads();

    {
      d_type *temp_ptr = working_data;
      working_data = sorted_data;
      sorted_data = temp_ptr;
    }

    {
      unsigned *temp_ptr = indexes;
      indexes = sorted_indexes;
      sorted_indexes = temp_ptr;
    }
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

template <typename T> void fill(T *arr, unsigned size, T val) {
  for (unsigned i = 0; i < size; ++i) {
    arr[i] = val;
  }
}

inline double average_range(std::vector<double> vals, unsigned start,
                        unsigned end) {
  double sum = 0;
  for (unsigned i = start; i < end; ++i) {
    sum += vals[i];
  }
  return sum / (end - start);
}

int main() {
  int ngpus = 0;
  cudaGetDeviceCount(&ngpus);
  printf("ngpus = %d\n", ngpus);
  if (ngpus > 0)
    cudaSetDevice(0);
  else
    return 0;
  int num_threads = omp_get_max_threads();
  printf("threads = %d\n", num_threads);

  unsigned m = 1024;
  unsigned n = 1024;

  std::vector<d_type> range_to_m;

  for (unsigned i = 0; i < m; ++i) {
    range_to_m.push_back(i);
  }

  d_type **data;
  loc_value **data_struct;
  d_type **max_vals;
  unsigned **max_locs;
  unsigned **cpu_max_locs;

  cudaMallocManaged(&data, n * sizeof(d_type *));
  cudaMallocManaged(&data_struct, n * sizeof(loc_value *));
  cudaMallocManaged(&max_vals, n * sizeof(d_type *));
  cudaMallocManaged(&max_locs, n * sizeof(unsigned *));
  cpu_max_locs = new unsigned *[n];

  cudaMallocManaged(&data[0], n * m * sizeof(d_type));
  cudaMallocManaged(&data_struct[0], n * m * sizeof(loc_value));
  cudaMallocManaged(&max_vals[0], n * 2 * sizeof(d_type));
  cudaMallocManaged(&max_locs[0], n * 2 * sizeof(unsigned));
  cpu_max_locs[0] = new unsigned[n * 2];

  for (unsigned i = 1; i < n; ++i) {
    data[i] = data[0] + i * m;
    data_struct[i] = data_struct[0] + i * m;
    max_vals[i] = max_vals[0] + i * 2;
    max_locs[i] = max_locs[0] + i * 2;
    cpu_max_locs[i] = cpu_max_locs[0] + i * 2;
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

  unsigned nth_max = static_cast<unsigned>(std::rand()) % m;

  unsigned iterations = 10;

  std::vector<double> cpu_time_nth;
  std::vector<double> cpu_time_max;
  std::vector<double> gpu_time_nth_and_max;
  std::vector<double> gpu_time_max;
  std::vector<double> gpu_time_max_warp_per_n;


  /* ---------------  TASK 1  ------------ */

  {
    fill(cpu_max_locs[0], n * 2, static_cast<unsigned>(-1));
    fill(max_vals[0], n * 2, static_cast<d_type>(0));


    for (unsigned iter = 0; iter < iterations; ++iter) {
      auto t1 = h_clock::now();
      #pragma omp parallel for
      for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < m; ++j) {
          if (data[i][j] > max_vals[i][0]) {
            max_vals[i][0] = data[i][j];
            cpu_max_locs[i][0] = j;
          }
        }
      }
      auto t2 = h_clock::now();

      cpu_time_max.push_back(
          chr::duration_cast<chr::duration<double>>(t2 - t1).count());
    }

    for (unsigned iter = 0; iter < iterations; ++iter) {
      auto t1 = h_clock::now();

      #pragma omp parallel for
      for (unsigned i = 0; i < n; ++i) {
        auto selected = quick_select(data_struct[i], 0, m, nth_max);
        max_vals[i][1] = selected.val;
        cpu_max_locs[i][1] = selected.idx;
      }
      auto t2 = h_clock::now();

      cpu_time_nth.push_back(
          chr::duration_cast<chr::duration<double>>(t2 - t1).count());
    }

    for (unsigned i = 0; i < n; ++i) {
      assert(max_vals[i][0] == m - 1);
      assert(max_vals[i][1] == nth_max);
    }

    std::cout << "==== cpu passed tests ====" << std::endl;
  }

  /* ---------------  TASK 2  ------------ */

  {
    {
      fill(max_locs[0], n * 2, static_cast<unsigned>(-1));
      fill(max_vals[0], n * 2, static_cast<d_type>(-1));

      for (unsigned iter = 0; iter < iterations; ++iter) {
        auto t1 = h_clock::now();

        get_max_warp_per_n<<<n, 32>>>(data_struct, max_vals, max_locs, m);

        cuda_error_chk(cudaDeviceSynchronize());

        auto t2 = h_clock::now();

        gpu_time_max_warp_per_n.push_back(
            chr::duration_cast<chr::duration<double>>(t2 - t1).count());
      }

      for (unsigned i = 0; i < n; ++i) {
        assert(max_vals[i][0] == m - 1);
        assert(max_locs[i][0] == cpu_max_locs[i][0]);
      }
    }
    {
      fill(max_locs[0], n * 2, static_cast<unsigned>(-1));
      fill(max_vals[0], n * 2, static_cast<d_type>(-1));

      for (unsigned iter = 0; iter < iterations; ++iter) {
        auto t1 = h_clock::now();

        get_max<<<n, m, (m - 1 + WARP_SIZE) / WARP_SIZE * sizeof(loc_value)>>>(
            data_struct, max_vals, max_locs, m);

        cuda_error_chk(cudaDeviceSynchronize());

        auto t2 = h_clock::now();

        gpu_time_max.push_back(
            chr::duration_cast<chr::duration<double>>(t2 - t1).count());
      }

      for (unsigned i = 0; i < n; ++i) {
        assert(max_vals[i][0] == m - 1);
        assert(max_locs[i][0] == cpu_max_locs[i][0]);
      }
    }
  }

  /* ---------------  TASK 3  ------------ */

  {
    fill(max_locs[0], n * 2, static_cast<uint32_t>(-1));
    fill(max_vals[0], n * 2, static_cast<d_type>(-1));

    uint32_t num_threads_per_block = m / 4;
    /* uint32_t num_threads_per_block = 32; */
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

    unsigned shared_count = num_warps_per_n * BINS_PER_PASS + num_warps_per_n +
                          +num_warps_per_n * m_iterations + m * 4 +
                          BINS_PER_PASS;

    for (unsigned iter = 0; iter < iterations; ++iter) {
      auto t1 = h_clock::now();

      sort_maxes<<<n, num_threads_per_block, shared_count * sizeof(uint32_t)>>>(
          data, max_vals, max_locs, m, nth_max, num_warps_per_n);

      cuda_error_chk(cudaDeviceSynchronize());

      auto t2 = h_clock::now();

      gpu_time_nth_and_max.push_back(
          chr::duration_cast<chr::duration<double>>(t2 - t1).count());
    }

    for (unsigned i = 0; i < n; ++i) {
      assert(max_vals[i][0] == m - 1);
      assert(max_vals[i][1] == nth_max);
      assert(max_locs[i][0] == cpu_max_locs[i][0]);
      assert(max_locs[i][1] == cpu_max_locs[i][1]);
    }

    std::cout << "==== gpu passed tests ====" << std::endl;
  }

  unsigned start = 2;

  std::cout << "cpu_time_nth: " << average_range(cpu_time_nth, start, iterations)
            << "\n"
            << "cpu_time_max: " << average_range(cpu_time_max, start, iterations)
            << "\n"
            << "gpu_time_nth_and_max: "
            << average_range(gpu_time_nth_and_max, start, iterations) << "\n"
            << "gpu_time_max_warp_per_n: "
            << average_range(gpu_time_max_warp_per_n, start, iterations) << "\n"
            << "gpu_time_max: " << average_range(gpu_time_max, start, iterations)
            << std::endl;

  cudaFree(data[0]);
  cudaFree(data);

  cudaFree(data_struct[0]);
  cudaFree(data_struct);

  cudaFree(max_vals[0]);
  cudaFree(max_vals);

  cudaFree(max_locs[0]);
  cudaFree(max_locs);

  delete[] cpu_max_locs;

  return 0;
}
