#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=0:03:00
#SBATCH -p apma2822

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=2
#SBATCH --tasks-per-node=8

# Specify a job name:
#SBATCH -J lecture_5

# Specify an output file
#SBATCH -o lecture_5-%j.out
#SBATCH -e lecture_5-%j.out

# Run a command

srun --mpi=pmi2 ./bin/app

