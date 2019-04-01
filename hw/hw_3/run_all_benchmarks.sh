#!/bin/bash

sbatch -J hw_3_2  -n 2 --nodes=1 -o 2_process.out  -e 2_process_error.out  run_benchmark.sh
sbatch -J hw_3_3  -n 3 --nodes=1 -o 3_process.out  -e 3_process_error.out  run_benchmark.sh
sbatch -J hw_3_4  -n 4 --nodes=1 -o 4_process.out  -e 4_process_error.out  run_benchmark.sh
sbatch -J hw_3_5  -n 5 --nodes=1 -o 5_process.out  -e 5_process_error.out  run_benchmark.sh
sbatch -J hw_3_8  -n 8 --nodes=1 -o 8_process.out  -e 8_process_error.out  run_benchmark.sh
sbatch -J hw_3_16 -n 16 --nodes=2 -o 16_process.out -e 16_process_error.out run_benchmark.sh
sbatch -J hw_3_24 -n 24 --nodes=2 -o 24_process.out -e 24_process_error.out run_benchmark.sh
