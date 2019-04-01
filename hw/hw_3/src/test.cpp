#include <vector>
#include <mpi.h>
#include <omp.h>
#include "matrix.h"

struct TestParams
{
  int size_i;
  int size_j;
  int size_k;
  bool use_mkl;
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int omp_num_threads = omp_get_max_threads();
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


  //test on small matrices
  std::vector<TestParams> tests = {
    {240, 60, 120, false},
    {240, 60, 120, true},
    {240, 120, 60, false},
    {240, 120, 60, true},
    {120, 60, 240, false},
    {120, 60, 240, true},
    {120, 240, 60, false},
    {120, 240, 60, true},
    {60, 240, 120, false},
    {60, 240, 120, true},
    {60, 120, 240, false},
    {60, 120, 240, true}};

  auto f_a = [](int i, int j) { return i * 0.3 + j * 0.4; };
  auto f_b = [](int j, int k) { return j * 0.5 + k * 0.6; };

  for(auto &test : tests) {
    distributed_matrix_multiply(test.size_i, test.size_j, test.size_k,
        world_rank, world_size, omp_num_threads, f_a, f_b, test.use_mkl, true);
  }

  std::cout << "finished all tests" << std::endl;
}
