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
    } else {

      auto t1 = h_clock::now();
      method.run();
      auto t2 = h_clock::now();
      time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
                 .count();
    }
    times[i] = time;
  }
}

void allocate_tex_object(cudaTextureObject_t &tex, double *dev_ptr, int n) {

  cudaTextureDesc td;
  memset(&td, 0, sizeof(td));
  td.normalizedCoords = 0;
  td.addressMode[0] = cudaAddressModeClamp;
  td.readMode = cudaReadModeElementType;

  struct cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeLinear;
  res_desc.res.linear.devPtr = dev_ptr;
  res_desc.res.linear.sizeInBytes = n * sizeof(double);
  res_desc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
  res_desc.res.linear.desc.x = 32;
  res_desc.res.linear.desc.y = 32;

  cuda_error_chk(cudaCreateTextureObject(&tex, &res_desc, &td, NULL));
}
