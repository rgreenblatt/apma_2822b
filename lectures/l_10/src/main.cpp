#include <omp.h>
#include <iostream>
#include <math.h>

int main(int argc, char *argv[])
{
  double *x, *y;
  int N = 4000000;

  x = new double[N];
  y = new double[N];

  for (int i = 0; i < N; ++i) {
    x[i] = 0.1 * i;
  }

  #pragma omp target enter data map(to:x[0:N]) map(alloc:y[0:N])


  #pragma omp target teams distribute parallel for
  for (int i = 0; i < N; ++i) {
    y[i] = sin(x[i]) * cos(x[i]);
  }

  return 0;
}
