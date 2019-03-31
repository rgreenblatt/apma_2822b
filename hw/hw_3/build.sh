#!/bin/bash

CXX=icc cmake -H. -Bbuild

CXX=icc cmake --build build -- -j3
