# DTensor v2.0 - Distributed Tensor Framework

A high-performance C++ framework for tensor parallelism in distributed deep learning. DTensor implements efficient sharding strategies and distributed operations using MPI and NCCL for multi-GPU training.

## 🚀 Features

### Core Capabilities

- **Tensor Parallelism**: Column-parallel and row-parallel matrix multiplication
- **Distributed Collectives**: AllReduce, AllGather, ReduceScatter, Broadcast via NCCL
- **GPU Acceleration**: cuBLAS-optimized matrix operations
- **Memory Management**: Custom caching allocator for efficient GPU memory reuse (~100x speedup)
- **Flexible Sharding**: Support for replicated and sharded tensor layouts

### Distributed Operations

- **Column-Parallel MatMul**: Split weight columns across GPUs (no communication)
- **Row-Parallel MatMul**: Split weight rows with AllReduce synchronization
- **Layout Management**: Automatic tracking of global/local tensor shapes
- **Stream-based Execution**: Asynchronous GPU operations with CUDA streams

## 🏗️ Architecture

```
DTensor (Distributed Tensor Layer)
    ├── Layout Management (Sharding strategies)
    ├── Distributed Collectives (NCCL)
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
- Implements tensor parallel strategies

**ProcessGroup** - MPI/NCCL communication manager

- Handles collective operations
- Manages CUDA streams
- Provides asynchronous work tracking

**CachingAllocator** - GPU memory pool

- Reuses freed memory blocks
- Per-stream memory management
- Best-fit allocation strategy

**TensorLib** - Local tensor operations

- Element-wise operations
- Matrix multiplication
- Reductions and activations

## 📦 Installation

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

## 🎯 Usage

### Basic Example: Tensor Parallel MLP

```cpp
#include "tensor/dtensor.h"
#include "process_group/process_group.h"

// Initialize MPI and NCCL
MPI_Init(&argc, &argv);
int rank, world_size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

// Create process group and mesh
auto mesh = std::make_shared<Mesh>(world_size);
auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);

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

## 🧪 Testing

### Run MLP Forward Pass Test

```bash
mpirun -np 2 ./test_mlp_forward
```

**Expected Output:**

```
Layer 1 Output (Column-Sharded):
  Rank 0: [2, 2, 2, 2]  # Different on each rank
  Rank 1: [4, 4, 4, 4]

Layer 2 Output (Replicated):
  Both ranks: [24, 24, 24, 24]  # Identical after AllReduce
```

## 📊 Performance

### MLP Forward Pass (2 layers, batch=2, hidden=4)

- **Sequential**: Each rank computes full forward pass independently
- **Tensor Parallel**: Split computation across 2 GPUs

**Results:**

- ✅ Column-parallel: No communication overhead
- ✅ Row-parallel: Single AllReduce per layer
- ✅ Memory savings: Model parameters split across GPUs

### Memory Allocator Performance

- **Without caching**: ~200μs per allocation cycle (cudaMalloc + cudaFree)
- **With caching**: ~2μs per allocation cycle (cache hit)
- **Speedup**: ~100x faster memory operations

## 🔧 Configuration

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

### Tensor Parallel Patterns

**Column-Parallel**: `Y[M, N/P] = X[M, K] @ W[K, N/P]`

- No communication needed
- Preferred for first MLP layer

**Row-Parallel**: `Y[M, N] = X[M, K/P] @ W[K/P, N] + AllReduce`

- Requires AllReduce for final result
- Preferred for second MLP layer

## 🗺️ Roadmap

### Current Status

- ✅ Tensor parallelism (column/row parallel)
- ✅ Basic distributed operations
- ✅ MLP forward pass
- ✅ GPU memory management
- ✅ cuBLAS acceleration

### Planned Features

- [ ] Backward pass & gradients (autograd integration)
- [ ] Additional operations (ReLU, Softmax, LayerNorm)
- [ ] Pipeline parallelism
- [ ] Sequence parallelism
- [ ] Model checkpointing
- [ ] Python bindings

## 📚 Documentation

- [Caching Allocator Design](docs/caching_allocator_explained.md)
- [Tensor Ops Bridge](docs/tensor_ops_bridge_explained.md)

## 🤝 Contributing

This is an active research project. Contributions welcome!

### Team

- **DTensor Team**: Distributed tensor framework
- **Autograd Team**: Automatic differentiation (integration pending)
- **TensorLib Team**: Core tensor operations

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Inspired by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) tensor parallelism
- Uses NVIDIA NCCL for collective communications
- Built on custom TensorLib for local operations

---

**Status**: Active Development | **Version**: 2.0 | **Last Updated**: November 2025
