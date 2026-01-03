# DTensor v2.0 - Distributed Tensor Framework

A high-performance C++ framework for tensor parallelism in distributed deep learning. DTensor implements efficient sharding strategies and distributed operations using MPI and NCCL for multi-GPU training, with a focus on asynchronous execution and communication overlap.

## Features

### Core Capabilities

- **Tensor Parallelism**: Column-parallel and row-parallel matrix multiplication
- **Distributed Collectives**: AllReduce, AllGather, ReduceScatter, Broadcast via NCCL
- **Async Execution**: Multi-stream architecture for overlapping computation and communication
- **Memory Management**: Custom caching allocator for efficient GPU memory reuse (~100x speedup) and lazy allocation
- **Flexible Sharding**: Support for replicated and sharded tensor layouts (Row/Column)
- **Attention Layer**: Tensor-parallel Multi-Head Attention implementation

### Distributed Operations

- **Column-Parallel MatMul**: Split weight columns across GPUs (no communication)
- **Row-Parallel MatMul**: Split weight rows with AllReduce synchronization
- **Lazy Partial Reductions**: Defer reductions in row-parallel operations for better performance
- **Layout Management**: Automatic tracking of global/local tensor shapes and transitions
- **GPU-Native Initialization**: Efficient scatter/broadcast from root rank

## Architecture

```
DTensor (Distributed Tensor Layer)
    ├── Layout Management (Sharding strategies)
    ├── Distributed Collectives (ProcessGroupNCCL)
    └── Local Tensor Operations
            ↓
    TensorOpsBridge (Abstraction Layer)
            ↓
    TensorLib (Core Tensor Implementation)
            ↓
    cuBLAS / CUDA Kernels
```

### Key Components

**DTensor** - Distributed tensor with sharding awareness

- Manages global and local tensor shapes
- Coordinates communication across GPUs
- Implements tensor parallel strategies (MatMul, Attention)
- Handles layout transitions (Redistribute, Shard, Replicate)

**ProcessGroupNCCL** - NCCL communication manager

- Handles collective operations (AllReduce, AllGather, etc.)
- Manages CUDA streams (Compute, Comm, Data streams)
- Provides asynchronous work tracking and event synchronization

**CachingAllocator** - GPU memory pool

- Reuses freed memory blocks
- Per-stream memory management
- Best-fit allocation strategy

**TensorLib** - Local tensor operations

- Element-wise operations
- Matrix multiplication
- Reductions and activations

## Installation

### Prerequisites

- CUDA Toolkit (>= 11.0)
- NVIDIA NCCL (>= 2.0)
- OpenMPI or MPICH
- GCC/G++ with C++17 support
- cuBLAS library

### Building

```bash
# Clone the repository
git clone https://github.com/skywxlker-ops/TensorParallelismBeta.git
cd DTensor_v2.0

# Build TensorLib (required)
cd Tensor-Implementations
make
cd ..

# Build DTensor
make

# Run tests
make test_mlp_forward
mpirun -np 2 ./test_mlp_forward
```

## Usage

### Basic Example: Tensor Parallel MLP

```cpp
#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"

// Initialize MPI and NCCL
MPI_Init(&argc, &argv);
int rank, world_size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

// Create process group and mesh
auto mesh = std::make_shared<DeviceMesh>(world_size);
auto pg = std::make_shared<ProcessGroupNCCL>(rank, world_size, nccl_id);

// Layer 1: Column-Parallel (X @ W1)
// Input: Replicated [batch, hidden]
// Weight: Column-sharded [hidden, intermediate/world_size]
DTensor X(mesh, pg);
X.setData(input_data, Layout(mesh, {batch, hidden}, ShardingType::REPLICATED));

DTensor W1(mesh, pg);
W1.setData(w1_data, Layout(mesh, {hidden, intermediate}, ShardingType::SHARDED, 1));

DTensor Y1 = X.matmul(W1);  // No communication!

// Layer 2: Row-Parallel (Y1 @ W2)
// Input: Column-sharded from Y1
// Weight: Row-sharded [intermediate/world_size, hidden]
DTensor W2(mesh, pg);
W2.setData(w2_data, Layout(mesh, {intermediate, hidden}, ShardingType::SHARDED, 0));

DTensor Y2 = Y1.matmul(W2);  // AllReduce inside
```

## Testing

### Run MLP Forward Pass Test

```bash
mpirun -np 2 ./test_mlp_forward
```

### Run Attention Layer Test

```bash
mpirun -np 2 ./test_attention_parallel
```

### Run Async Timing Analysis

```bash
mpirun -np 2 ./mlp_testing/test_mlp_async_timing
```

## Performance

### Optimizations

- **Multi-Stream Execution**: Overlaps compute (MatMul) with communication (AllReduce).
- **Lazy Allocation**: Temporary buffers are allocated only when needed and reused.
- **Async Collectives**: Non-blocking NCCL calls allow CPU to look ahead.

### Memory Allocator

- **Without caching**: ~200μs per allocation cycle
- **With caching**: ~2μs per allocation cycle (cache hit)
- **Speedup**: ~100x faster memory operations

## Configuration

### Sharding Strategies

**REPLICATED** - Tensor copied on all ranks

```cpp
Layout(mesh, {M, N}, ShardingType::REPLICATED)
```

**SHARDED** - Tensor split along dimension

```cpp
Layout(mesh, {M, N}, ShardingType::SHARDED, dim)
// dim=0: Row-sharded
// dim=1: Column-sharded
```

## Roadmap

### Current Status

- ✅ Tensor parallelism (column/row parallel)
- ✅ Distributed collectives (NCCL)
- ✅ MLP forward pass
- ✅ Attention layer (Forward)
- ✅ GPU memory management & Caching Allocator
- ✅ Async execution & Stream overlap
- ✅ Lazy Partial Matmul
- 🚧 Backward pass & Autograd (In Progress)

### Planned Features

- [ ] Pipeline parallelism
- [ ] Sequence parallelism
- [ ] Model checkpointing
- [ ] Python bindings (PyBind11)

## Contributing

This is an active research project. Contributions welcome!

### Team

- **DTensor Team**: Distributed tensor framework
- **Autograd Team**: Automatic differentiation
- **TensorLib Team**: Core tensor operations

## Acknowledgments

- Inspired by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) tensor parallelism
- Uses NVIDIA NCCL for collective communications
- Built on custom TensorLib for local operations

---

**Status**: Active Development | **Version**: 2.0 | **Last Updated**: January 2026
