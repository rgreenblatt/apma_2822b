#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -t 00:05:00
#SBATCH -o hw_4.out

# Load CUDA module
module load cuda

# Compile CUDA program and run
./clean.sh
./build.sh
./bin/app
