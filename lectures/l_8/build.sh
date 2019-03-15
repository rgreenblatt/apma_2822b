#!/bin/bash

CC=gcc-6 CXX=g++-6 cmake -H. -Bbuild -DCUDA_HOST_COMPILER=/usr/bin/gcc-6

CC=gcc-6 CXX=g++-6 cmake --build build -- -j3
