#!/bin/bash

#SBATCH --time=0:10:00
#SBATCH -p batch

# Run a command

srun --mpi=pmi2 ./bin/app
