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
`DTensor` is a high-level wrapper around a local GPU `Tensor`.

**Constructor Parameters:**
- `name`: A string identifier for the tensor, useful for debugging and tracking memory allocations.
- `sd`: Standard deviation for random initialization (default: `0.02f`).
- `seed`: Random seed for initialization (default: `42`). Using the same seed across ranks ensures identical initialization for replicated parameters.

```cpp
DTensor h(mesh, pg, in_layout, "Input_Tensor", 0.02f, 42);
```

## How to use Tensor Parallelism in a Script

To implement Tensor Parallelism (Megatron-LM style), follow these steps:

### 1. Layer Configuration

#### Column Parallel (`DColumnLinear`)
Shards the output dimension across GPUs.
- `use_bias`: Boolean (`true`/`false`) to enable/disable bias.
- `sd` & `seed`: Control weight initialization.

```cpp
dnn::DColumnLinear fc1(mesh, pg, B, T, C, F, {}, true, 0.02f, 42);
```

#### Row Parallel (`DRowLinear`)
Shards the input dimension across GPUs and typically synchronizes the output.
- `use_bias`: Boolean to enable/disable bias.
- `sync_output`: Boolean (`true`/`false`). If `true`, performs an `all-reduce` on the output to make it replicated/consistent across ranks. By default, it is often `true` for final layers of a block.

```cpp
dnn::DRowLinear fc4(mesh, pg, B, T, F, C, {}, true, 0.02f, 42, true);
```

#### Language Model Head (`DLMHead`)
- `use_tied_weights`: Boolean (`true`/`false`). If `true`, the layer shares weights with an embedding layer (typically the token embedding).

```cpp
// Using tied weights
dnn::DLMHead lm_head(mesh, pg, B, T, C, V, true, wte.weight.get());

// Without tied weights
dnn::DLMHead lm_head(mesh, pg, B, T, C, V, false);
```

### 2. Automatic Synchronization
When using `dnn::` blocks, the library handles the complexity:
- `DColumnLinear` performs a local matmul.
- `DRowLinear` performs a local matmul followed by an `all-reduce` if `sync_output` is `true`.

### 3. Training Loop Integration
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
// Handled automatically by backward hooks in DTensor for sharded params.
// For replicated params (like LayerNorm), manual sync is needed:
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
