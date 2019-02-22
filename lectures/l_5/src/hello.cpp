#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    int N = 100;
    double *parameters = new double[N];

    if(world_rank == 0) {
        #pragma omp for
        for(int i = 0; i < N; i++) {
          parameters[i] = i * 0.5;
        }
    }

    MPI_Bcast(parameters, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors %f val\n",
           processor_name, world_rank, world_size, parameters[world_rank]);

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
