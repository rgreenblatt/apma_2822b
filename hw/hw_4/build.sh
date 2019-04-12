#!/bin/bash

cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DCMAKE_CXX_COMPILER=/usr/bin/g++-6 \
  cmake -H. -Bbuild

cmake --build build -- -j3
