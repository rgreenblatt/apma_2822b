#!/bin/bash

sbatch -J hw_3_16_1 -n 16  --nodes=1 -o 16_1.out  -e 16_1_error.out  run_benchmark.sh
sbatch -J hw_3_32_1 -n 32  --nodes=2 -o 32_1.out  -e 32_1_error.out  run_benchmark.sh
sbatch -J hw_3_64_1 -n 64  --nodes=4 -o 64_1.out  -e 64_1_error.out  run_benchmark.sh

sbatch -J hw_3_1_8  -c 8   --nodes=1 -o 1_8.out   -e 1_8_error.out   run_benchmark.sh
sbatch -J hw_3_1_16 -c 16  --nodes=1 -o 1_16.out  -e 1_16_error.out  run_benchmark.sh
sbatch -J hw_3_2_8  -c 8   --nodes=2 -o 2_8.out   -e 2_8_error.out   run_benchmark.sh
sbatch -J hw_3_2_16 -c 16  --nodes=2 -o 2_16.out  -e 2_16_error.out  run_benchmark.sh
sbatch -J hw_3_4_8  -c 8   --nodes=4 -o 4_8.out   -e 4_8_error.out   run_benchmark.sh
sbatch -J hw_3_4_16 -c 16  --nodes=4 -o 4_16.out  -e 4_16_error.out  run_benchmark.sh

sbatch -J hw_3_8_8  -n 8 -c 8   --nodes=4 -o 8_8.out   -e 8_8_error.out   run_benchmark.sh
sbatch -J hw_3_16_4  -n 16 -c 4   --nodes=4 -o 16_4.out   -e 16_4_error.out   run_benchmark.sh
