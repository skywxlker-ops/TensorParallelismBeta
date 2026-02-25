#!/bin/bash
set -e

echo "Compiling Standalone Sigmoid Test..."
g++ -Iinclude -I/usr/local/cuda/include -DWITH_CUDA -std=c++20 -fPIC -Wall -Wextra -O3 -g -fopenmp \
    Tests/autograd_operation_testing/test_sigmoid_standalone.cpp \
    -o sigmoid_benchmark_runner \
    -L/usr/local/cuda/lib64 -Llib \
    -Xlinker -rpath -Xlinker '$ORIGIN/lib' \
    -ltensor -lcudart -ltbb -lcurand -lcublas

echo "Running Sigmoid Benchmark..."
./sigmoid_benchmark_runner
