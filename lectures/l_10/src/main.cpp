#include <omp.h>
#include <iostream>

int main(int, char *[])
{
  double *x, *y;
  const unsigned long N = 4000000;

  x = new double[N];
  y = new double[N];

  for (unsigned long i = 0; i < N; ++i) {
    x[i] = 0.1 * i;
  }

  #pragma omp target enter data map(to:x[0:N]) map(alloc:y[0:N])


  #pragma omp target teams distribute parallel for
  for (unsigned long i = 0; i < N; ++i) {
    y[i] = (x[i] - 3.0) * (x[i] + 10.0);
  }

  return 0;
}
