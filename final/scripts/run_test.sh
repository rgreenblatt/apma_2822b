#!/bin/bash

#SBATCH --time=0:10:00
#SBATCH -p apma2822

# Run a command

srun --mpi=pmi2 ./bin/miniFE
