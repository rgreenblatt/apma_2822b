#!/bin/bash

#SBATCH --time=0:05:00
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH -c 16

#SBATCH -o measure_statistics.out 
#SBATCH -e measure_statistics_error.out

srun --mpi=pmi2 ./bin/measure_statistics
