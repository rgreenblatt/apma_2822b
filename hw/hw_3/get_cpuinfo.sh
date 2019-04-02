#!/bin/bash

#SBATCH --time=0:01:00
#SBATCH -p batch
#SBATCH --nodes=2

# Run a command

#SBATCH -o cpuinfo.out 
#SBATCH -e cpuinfo_error.out

cat /proc/cpuinfo
