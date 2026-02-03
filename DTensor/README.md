# DTensor v2.0: N-Dimensional Tensor Parallelism

This repository contains a C++ implementation of Distributed Tensors (DTensor) with support for n-dimensional parallelism, similar to PyTorch's `DeviceMesh` and `DTensor` abstractions.

## Features
- **DeviceMesh**: N-dimensional device topology management.
- **Placements**: `Shard`, `Replicate`, `Partial` support.
- **DTensor**: Distributed tensor wrapper around local tensors.
- **Collectives**: Efficient NCCL-based communication.
- **Benchmarks**: Tools to measure communication and compute performance.

## Prerequisites
- MPI (OpenMPI)
- CUDA Toolkit
- NCCL
- C++17 compiler

## Building
Run `make` to build the library and tests:
```bash
cd DTensor_v2.0
make
```

## Running Tests

### 1. Test DeviceMesh & Placements
Verifies the correctness of the n-dimensional mesh and placement logic (Shard/Replicate).
```bash
make test_device_mesh
mpirun -np 2 ./tests/test_device_mesh
```

### 2. MatMul Benchmark (Row vs Column Parallel)
Measures the performance (Latency & TFLOPS) of Column-Parallel (compute-bound) vs Row-Parallel (communication-bound) matrix multiplication across various sizes.
```bash
make matmul_benchmark
mpirun -np 2 ./benchmarks/matmul_benchmark
```

### 3. NCCL Collectives Benchmark
Measures the raw latency and bandwidth of NCCL operations (AllReduce, AllGather, etc.) for different data sizes.
```bash
make nccl_benchmark
mpirun -np 2 ./benchmarks/nccl_benchmark
```

## Directory Structure
- `tensor/`: Core DTensor, DeviceMesh, and Layout implementations.
- `process_group/`: NCCL process group management.
- `bridge/`: Bridge between DTensor and local tensor operations.
- `tests/`: Unit and integration tests.
- `benchmarks/`: Performance benchmarking tools.
