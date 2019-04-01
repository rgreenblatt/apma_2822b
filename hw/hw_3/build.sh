#!/bin/bash


MKLROOT=/opt/intel/system_studio_2019/compilers_and_libraries_2019/linux/mkl/ CXX=icc cmake -H. -Bbuild

MKLROOT=/opt/intel/system_studio_2019/compilers_and_libraries_2019/linux/mkl/ CXX=icc cmake --build build -- -j3
