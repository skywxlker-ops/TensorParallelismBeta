# TensorParallelismBeta

A high-performance C++/CUDA library for distributed Deep Learning, specifically designed for **Tensor Parallelism (TP)** without heavy dependencies like PyTorch.

## Core Concepts

### 1. DeviceMesh
The `DeviceMesh` defines the 2D/3D grid of GPUs. For 1D Tensor Parallelism, it represents a simple group of $N$ GPUs.

```cpp
std::vector<int> ranks(world_size);
std::iota(ranks.begin(), ranks.end(), 0);
DeviceMesh mesh({world_size}, ranks);
auto pg = mesh.get_process_group(0);
```

### 2. Layout
The `Layout` defines how a global tensor is distributed across the `DeviceMesh`.
- **Replicated**: Every GPU has the full copy of the tensor.
- **Sharded**: The tensor is split along a specific dimension (e.g., Row-parallel or Column-parallel).

```cpp
// Replicated Layout
Layout replicated_layout(mesh, {Batch, SeqLen, Hidden});

// Sharded Layout (Sharded on Dimension 2 - Column Parallel)
Layout sharded_layout(mesh, {Batch, SeqLen, Hidden}, 2);
```

### 3. DTensor (Distributed Tensor)
`DTensor` is a high-level wrapper around a local GPU `Tensor`. It manages:
- **Metadata**: Global shape and current layout.
- **Collectives**: Handles `all-reduce`, `all-gather`, and `shard` operations.
- **Autograd**: Automatically registers backward hooks for gradient synchronization.

## How to use Tensor Parallelism in a Script

To implement Tensor Parallelism (Megatron-LM style), follow these steps:

### 1. Define Layouts
For a linear layer $Y = XW + b$:
- **Column Parallel**: Shard $W$ along columns (dim 1). $Y$ will be sharded along dim 2.
- **Row Parallel**: Shard $W$ along rows (dim 0). $Y$ needs an `all-reduce` to become replicated.

### 2. Wrap Tensors in DTensor
Initialize your weights as `DTensor` with the appropriate sharding.

```cpp
// Column Linear: Shards on columns
dnn::DColumnLinear fc1(mesh, pg, B, T, C, F);

// Row Linear: Shards on rows, syncs output
dnn::DRowLinear fc4(mesh, pg, B, T, F, C);
```

### 3. Automatic Synchronization
When using `dnn::` blocks, the library handles the complexity:
- `DColumnLinear` performs a local matmul.
- `DRowLinear` performs a local matmul followed by an `all-reduce` (`sync_w_autograd`).

### 4. Training Loop Integration
```cpp
// 1. Zero Gradients
optimizer.zero_grad();

// 2. Forward Pass
DTensor logits = model.forward(input);

// 3. Loss Calculation (Parallel Cross Entropy or Gathered)
Tensor loss = parallel_cross_entropy(logits, targets);

// 4. Backward Pass
loss.backward(&grad_scale);

// 5. Gradient Synchronization
// Handled automatically by sync_w_autograd hooks in DTensor
// For replicated params, manual sync is needed:
pg->all_reduce_async(grad_ptr, grad_ptr, count, Dtype::Float32, sum)->wait();

// 6. Step
optimizer.step();
```

## Building and Running

### Build
Use the provided `Makefile` in the `DTensor` directory.
```bash
cd DTensor
make gpt2_tp_test
```

### Run
Launch with `mpirun` for multiple GPUs.
```bash
mpirun -np 2 ./gpt2_tp_test
```

## Project Structure
- `DTensor/tensor/`: Core `DTensor`, `Layout`, and `DeviceMesh` implementation.
- `DTensor/Tensor-Implementations/include/nn/`: Distributed layers (`DColumnLinear`, `DRowLinear`, etc.).
- `DTensor/TrainingScripts/`: End-to-end training examples using Tensor Parallelism.
- `MyBlas/`: Optimized CUDA kernels for matrix operations.
