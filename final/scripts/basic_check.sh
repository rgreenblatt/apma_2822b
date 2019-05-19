#!/bin/bash

#SBATCH --time=0:10:00
#SBATCH -p apma2822
#SBATCH --ntasks=2
#SBATCH -p gpu --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -o final.out

# Run a command

module load cuda/10.0.130
module load mpi/mvapich2-2.3b_gcc

srun --mpi=pmi2 ./bin/miniFE -nx 300 -ny 300 -nz 300 &
nvidia-smi -l 1
