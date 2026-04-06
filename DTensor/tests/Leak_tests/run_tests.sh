#!/bin/bash
set -e

# Build the library to ensure it's up to date
echo "Building main library..."
make all -j20

echo "Compiling benchmark suite..."
g++ -Iinclude -I/usr/local/cuda/include -DWITH_CUDA -std=c++20 -fPIC -Wall -Wextra -O3 -g -fopenmp \
    Tests/Autograd_operation_testing/benchmark_main.cpp \
    Tests/Autograd_operation_testing/test_ops_activations.cpp \
    Tests/Autograd_operation_testing/test_ops_losses.cpp \
    Tests/Autograd_operation_testing/test_ops_layers.cpp \
    -o autograd_benchmark_runner \
    -L/usr/local/cuda/lib64 -Llib \
    -Xlinker -rpath -Xlinker '$ORIGIN/lib' \
    -ltensor -lcudart -ltbb -lcurand -lcublas

echo "Running benchmark..."
if [ "$1" == "--valgrind" ]; then
    echo "Using Valgrind..."
    valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./autograd_benchmark_runner
else
    ./autograd_benchmark_runner
fi
