#!/bin/bash

CXX=clang++ cmake -H. -BDebug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES
CXX=clang++ cmake -H. -Bbuild

CXX=clang++ cmake --build build -- -j3
