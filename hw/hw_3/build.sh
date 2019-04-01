#!/bin/bash


BLAS_DIR=/gpfs/runtime/opt/openblas/0.2.19/ cmake -H. -Bbuild

BLAS_DIR=/gpfs/runtime/opt/openblas/0.2.19 cmake --build build -- -j3
