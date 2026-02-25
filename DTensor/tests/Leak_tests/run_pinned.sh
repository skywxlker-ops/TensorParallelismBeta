#!/bin/bash
set -e

echo "Compiling Pinned Memory Benchmark..."
g++ -Iinclude -I/usr/local/cuda/include -DWITH_CUDA -std=c++20 -fPIC -Wall -Wextra -O3 -g -fopenmp \
    Tests/autograd_operation_testing/test_pinned_transfer.cpp \
    -o pinned_transfer_runner \
    -L/usr/local/cuda/lib64 -Llib \
    -Xlinker -rpath -Xlinker '$ORIGIN/lib' \
    -ltensor -lcudart -ltbb -lcurand -lcublas

echo "Running Benchmark..."
./pinned_transfer_runner
