#!/bin/bash

if [ -f "/usr/bin/gcc-6" ]; then
    cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DCMAKE_CXX_COMPILER=/usr/bin/g++-6 \
        cmake -H. -Bbuild
else
    cmake -H. -Bbuild
fi

cmake --build build -- -j3
