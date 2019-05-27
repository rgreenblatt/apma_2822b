#!/bin/bash

mpirun -np 1 ./bin/miniFE -nx 240 -ny 240 -nz 240
mpirun -np 1 ./bin/miniFE -nx 280 -ny 280 -nz 280

mpirun -np 2 ./bin/miniFE -nx 240 -ny 240 -nz 240
mpirun -np 2 ./bin/miniFE -nx 320 -ny 320 -nz 320
mpirun -np 2 ./bin/miniFE -nx 353 -ny 353 -nz 353

mpirun -np 3 ./bin/miniFE -nx 240 -ny 240 -nz 240
mpirun -np 3 ./bin/miniFE -nx 320 -ny 320 -nz 320
mpirun -np 3 ./bin/miniFE -nx 404 -ny 404 -nz 404

mpirun -np 4 ./bin/miniFE -nx 240 -ny 240 -nz 240
mpirun -np 4 ./bin/miniFE -nx 320 -ny 320 -nz 320
mpirun -np 4 ./bin/miniFE -nx 520 -ny 520 -nz 520
mpirun -np 4 ./bin/miniFE -nx 445 -ny 445 -nz 445

mpirun -np 5 ./bin/miniFE -nx 240 -ny 240 -nz 240
mpirun -np 5 ./bin/miniFE -nx 320 -ny 320 -nz 320
mpirun -np 5 ./bin/miniFE -nx 520 -ny 520 -nz 520
mpirun -np 5 ./bin/miniFE -nx 479 -ny 479 -nz 479

mpirun -np 6 ./bin/miniFE -nx 240 -ny 240 -nz 240
mpirun -np 6 ./bin/miniFE -nx 320 -ny 320 -nz 320
mpirun -np 6 ./bin/miniFE -nx 520 -ny 520 -nz 520
mpirun -np 6 ./bin/miniFE -nx 509 -ny 509 -nz 509

mpirun -np 7 ./bin/miniFE -nx 240 -ny 240 -nz 240
mpirun -np 7 ./bin/miniFE -nx 320 -ny 320 -nz 320
mpirun -np 7 ./bin/miniFE -nx 520 -ny 520 -nz 520
mpirun -np 7 ./bin/miniFE -nx 536 -ny 536 -nz 536

mpirun -np 8 ./bin/miniFE -nx 240 -ny 240 -nz 240
mpirun -np 8 ./bin/miniFE -nx 320 -ny 320 -nz 320
mpirun -np 8 ./bin/miniFE -nx 520 -ny 520 -nz 520
mpirun -np 8 ./bin/miniFE -nx 560 -ny 560 -nz 560
