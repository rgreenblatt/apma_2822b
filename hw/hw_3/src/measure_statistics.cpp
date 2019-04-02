#include "matrix.h"
#include <mpi.h>

namespace chr = std::chrono;
using h_clock = chr::high_resolution_clock;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  assert(world_size == 2);

  if (world_rank == 0) {
    int n = 4194304;

    double from[n];
    double to[n];

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      from[i] = i * 0.4;
    }

    auto t1 = h_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      to[i] = from[i];
    }
    auto t2 = h_clock::now();

    double time = chr::duration_cast<chr::duration<double>>(t2 - t1).count();

    double g_bytes_per_s = n / (1024. * 1024. * 1024.) * 8 * 2;
    std::cout << "======  within node transfer ======" << std::endl;
    std::cout << "total time: " << time
              << " gigabytes per second: " << g_bytes_per_s << std::endl
              << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  int n_mpi = 4194304;
  if (world_rank == 0) {

    double from[n_mpi];

    #pragma omp parallel for
    for (int i = 0; i < n_mpi; i++) {
      from[i] = i * 0.4;
    }

    auto t1 = h_clock::now();
    MPI_Send(from, n_mpi, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    auto t2 = h_clock::now();

    double time = chr::duration_cast<chr::duration<double>>(t2 - t1).count();

    double g_bytes_per_s = n_mpi / (1024. * 1024. * 1024.) * 8 * 2;
    std::cout << "======  between nodes transfer ======" << std::endl;
    std::cout << "total time: " << time
              << " gigabytes per second: " << g_bytes_per_s << std::endl;

  } else {

    double to[n_mpi];
    MPI_Status rec_status;
    MPI_Recv(to, n_mpi, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &rec_status);
  }

  return 0;
}
