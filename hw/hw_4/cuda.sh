#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 8 CPU cores
#SBATCH -n 8

#SBATCH -t 00:05:00
#SBATCH -o hw_04.out

# Load CUDA module
module load cuda/9.1.85.1

# Compile CUDA program and run
./clean.sh
./build.sh
nvprof -o app.nvpf ./bin/app
