#!/bin/bash

sbatch -J hw_3_16_1 -n 16  --nodes=1 -o 16_1.out  -e 16_1_error.out  run_benchmark.sh
sbatch -J hw_3_32_1 -n 32  --nodes=2 -o 32_1.out  -e 32_1_error.out  run_benchmark.sh
sbatch -J hw_3_48_1 -n 48  --nodes=3 -o 48_1.out  -e 48_1_error.out  run_benchmark.sh
sbatch -J hw_3_64_1 -n 64  --nodes=4 -o 64_1.out  -e 64_1_error.out  run_benchmark.sh

sbatch -J hw_3_1_8  -c 8   --nodes=1 -o 1_8_.out   -e 1_8__error.out  run_benchmark.sh
sbatch -J hw_3_1_16 -c 16  --nodes=1 -o 1_16_.out  -e 1_16_error.out  run_benchmark.sh
sbatch -J hw_3_2_8  -c 8   --nodes=2 -o 2_8_.out   -e 2_8__error.out  run_benchmark.sh
sbatch -J hw_3_2_16 -c 16  --nodes=2 -o 2_16_.out  -e 2_16_error.out  run_benchmark.sh
sbatch -J hw_3_3_8  -c 8   --nodes=3 -o 3_8_.out   -e 3_8__error.out  run_benchmark.sh
sbatch -J hw_3_3_16 -c 16  --nodes=3 -o 3_16_.out  -e 3_16_error.out  run_benchmark.sh
sbatch -J hw_3_4_8  -c 8   --nodes=4 -o 4_8_.out   -e 4_8__error.out  run_benchmark.sh
sbatch -J hw_3_4_16 -c 16  --nodes=4 -o 4_16_.out  -e 4_16_error.out  run_benchmark.sh
