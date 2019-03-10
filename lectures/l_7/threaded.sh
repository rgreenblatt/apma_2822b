#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=1:00:00
#SBATCH -p apma2822

# Default resources are 1 core with 2.8GB of memory per core.

# Use more cores (8):
#SBATCH -c 8

# Specify a job name:
#SBATCH -J lecture_3

# Specify an output file
#SBATCH -o lecture_3_%j.out
#SBATCH -e lecture_3_%j.out

# Run a command

./bin/app -threads 8
