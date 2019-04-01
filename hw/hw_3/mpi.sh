#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=0:10:00
#SBATCH -p apma2822

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=1
#SBATCH -n 4

# Specify a job name:
#SBATCH -J hw_3

# Specify an output file
#SBATCH -o hw_3-%j.out
#SBATCH -e hw_3-%j.out

# Run a command

srun --mpi=pmi2 ./bin/test
