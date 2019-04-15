#include "utils.h"
#include <assert.h>
#include <iostream>

void print_result(double *times, int iterations, const char *name) {
  printf("%s times: \nall times: ", name);
  for (int i = 0; i < iterations; i++) {
    printf("%e", times[i]);
    if (i != iterations - 1) {
      printf(",");
    }
  }
  assert(iterations > 2);
  double avg = 0.;
  for (int i = 2; i < iterations; i++) {
    avg += times[i];
  }
  avg /= (iterations - 2);

  printf("\naverage not including first two runs: %e\n\n", avg);
}

void time_function(int iterations, SpMvMethod &method, double *times,
                   bool is_cuda) {
  for (int i = 0; i < iterations; ++i) {
    double time;
    if (is_cuda) {
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      method.run();
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cuda_error_chk(cudaDeviceSynchronize());
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      time = milliseconds / 1000;
      /* cuda_error_chk(cudaDeviceSynchronize()); */
    } else {

      auto t1 = h_clock::now();
      method.run();
      auto t2 = h_clock::now();
      time =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
              .count();
    }
    times[i] = time;
  }
}
