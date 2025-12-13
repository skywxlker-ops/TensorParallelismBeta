# Learning Report - December 9, 2025

## Part 1: Understanding Tensor Parallelism - Column & Row Parallel MatMul

### Overview

Learned how tensor parallelism splits matrix operations across multiple GPUs in the MLP layer of transformer models.

**Key Concept:** Split weight matrices across GPUs to reduce memory and enable larger models.

---

### Background: Transformer MLP Architecture

Standard MLP uses **4x expansion**:

```
Input [B, T, C] → Linear1 [C, 4C] → Activation → Linear2 [4C, C] → Output [B, T, C]
```

Where:

- **B** = Batch size
- **T** = Token/sequence length  
- **C** = Embedding dimension

With P=2 GPUs, we split the 4C hidden dimension across GPUs (each gets 2C).

---

### Column-Parallel Matrix Multiplication

**Concept:** Split weight columns across GPUs. No communication needed!

```
GPU 0:                   GPU 1:
X [B*T, C] (full)        X [B*T, C] (full)
    ×                        ×
W1[C, 2C] (cols 0-2C)    W1[C, 2C] (cols 2C-4C)
    ↓                        ↓
H [B*T, 2C] (shard0)     H [B*T, 2C] (shard1)
```

**Code Implementation:**

```cpp
DTensor DTensor::_column_parallel_matmul(const DTensor& other) const {
    // X [B*T, C] @ W1 [C, 4C/P] → H [B*T, 4C/P]
    
    // Step 1: Local matmul - Full input × Weight shard
    OwnTensor::Tensor Y_shard = TensorOpsBridge::matmul(
        this->tensor_,           // X: replicated [B*T, C]
        other.local_tensor()     // W1 shard: [C, 4C/P]
    );
    
    // Step 2: Global shape if we gathered all shards
    std::vector<int> Y_global_shape = {
        this->layout_.get_global_shape()[0],      // B*T
        other.get_layout().get_global_shape()[1]  // 4C (not 4C/P!)
    };
    
    // Step 3: Mark output as column-sharded
    Layout Y_layout(device_mesh_, Y_global_shape, ShardingType::SHARDED, 1);
    return DTensor(device_mesh_, pg_, Y_shard, Y_layout);
}
```

**No `sync()` call = No communication!**

---

### Row-Parallel Matrix Multiplication

**Concept:** Each GPU computes partial result, then AllReduce sums them.

```
GPU 0:                      GPU 1:
H [B*T, 2C] (shard0)        H [B*T, 2C] (shard1)
    ×                           ×
W2[2C, C] (rows 0-2C)       W2[2C, C] (rows 2C-4C)
    ↓                           ↓
Y_partial0 [B*T, C]         Y_partial1 [B*T, C]
    └────────AllReduce(SUM)────────┘
              Y_final [B*T, C]
```

**Code Implementation:**

```cpp
DTensor DTensor::_row_parallel_matmul(const DTensor& other) const {
    // H [B*T, 4C/P] @ W2 [4C/P, C] → Y [B*T, C]
    
    // Step 1: Local matmul produces PARTIAL result
    OwnTensor::Tensor Y_partial = TensorOpsBridge::matmul(
        this->tensor_,           // H shard: [B*T, 4C/P]
        other.local_tensor()     // W2 shard: [4C/P, C]
    );
    
    // Step 2: Create replicated layout (all GPUs will have same data after sync)
    std::vector<int> Y_global_shape = {
        this->layout_.get_global_shape()[0],      // B*T
        other.get_layout().get_global_shape()[1]  // C
    };
    Layout Y_layout = Layout::replicated(device_mesh_, Y_global_shape);
    DTensor Y_out(device_mesh_, pg_, Y_partial, Y_layout);
    
    // Step 3: Sum partial results across GPUs
    Y_out.sync();  // AllReduce(SUM)
    
    return Y_out;
}
```

**`sync()` performs AllReduce = Communication required!**

---

### Key Differences

| Aspect | Column-Parallel | Row-Parallel |
|--------|----------------|--------------|
| Input | Replicated | Sharded |
| Weight split | By columns | By rows |
| Local result | Complete shard | Partial result |
| Communication | **None** | **AllReduce SUM** |
| Output | Sharded | Replicated |

---

### Communication Cost Analysis

For MLP with 2 GPUs:

- **Column-parallel:** 0 bytes
- **Row-parallel:** 2 × B×T×C × sizeof(float) bytes

**Total:** 1 AllReduce for entire MLP forward pass!

---

## Part 2: Layout Transformations Benchmark

### What I Built

Created a benchmark to measure how fast DTensor can redistribute data across GPUs. This is critical because different layers need different sharding patterns.

**File:** `tests/test_benchmark_layouts.cpp`

---

### Four Transformation Types

```cpp
// 1. Replicated → Row-sharded
DTensor X(mesh, pg);
X.setData(data, replicated_layout);
X.shard(0);  // Shard along dimension 0 (rows)

// 2. Replicated → Column-sharded  
X.shard(1);  // Shard along dimension 1 (columns)

// 3. Row-sharded → Column-sharded (redistribute)
X.replicate(0);  // AllGather to replicate
X.shard(1);      // Then shard along columns

// 4. Sharded → Replicated
X.replicate(0);  // AllGather operation
```

**What happens under the hood:**

```
Replicated → Sharded:
┌────────────┐  Broadcast  ┌────────────┐
│ GPU 0      │ ──────────→ │ GPU 0      │
│ Full data  │             │ Shard 0    │
└────────────┘             └────────────┘
                           ┌────────────┐
                           │ GPU 1      │
                           │ Shard 1    │
                           └────────────┘

Sharded → Replicated:
┌────────────┐             ┌────────────┐
│ GPU 0      │  AllGather  │ GPU 0      │
│ Shard 0    │ ─────────→  │ Full data  │
└────────────┘             └────────────┘
┌────────────┐             ┌────────────┐
│ GPU 1      │             │ GPU 1      │
│ Shard 1    │             │ Full data  │
└────────────┘             └────────────┘
```

---

### Benchmark Configuration

- **Test Sizes:** 512×512, 1K×1K, 2K×2K, 4K×4K, 8K×8K
- **Iterations:** 100 warmup + 500 measurement (for stable averages)
- **GPUs:** 2 GPUs with NCCL
- **Metrics:** Time, bandwidth, memory

---

### Results

```
[Matrix size: 512x512]
  Replicated → Row-sharded | Time:    0.26 ms | BW:   3.8 GB/s
  Replicated → Col-sharded | Time:    0.26 ms | BW:   3.8 GB/s
  Row-shard → Col-shard    | Time:    0.38 ms | BW:   2.6 GB/s
  Sharded → Replicated     | Time:    0.25 ms | BW:   4.0 GB/s

[Matrix size: 4096x4096]
  Replicated → Row-sharded | Time:   12.3 ms | BW:   5.1 GB/s
  Replicated → Col-sharded | Time:   12.0 ms | BW:   5.2 GB/s
  Row-shard → Col-shard    | Time:   19.2 ms | BW:   3.3 GB/s
  Sharded → Replicated     | Time:   13.5 ms | BW:   4.6 GB/s

[Matrix size: 8192x8192]
  Replicated → Row-sharded | Time:   49.2 ms | BW:   5.1 GB/s
  Replicated → Col-sharded | Time:   49.2 ms | BW:   5.1 GB/s
```

---

### What Each Transformation Does

#### 1. Replicated → Row-sharded (0.26 ms)

**Before:**

```
GPU 0: X [512, 512] - Full matrix (REPLICATED)
GPU 1: X [512, 512] - Full matrix (REPLICATED)
```

**Operation:** `X.shard(0)` - Shard along dimension 0 (rows)

**After:**

```
GPU 0: X [256, 512] - Rows 0-255 (ROW-SHARDED)
GPU 1: X [256, 512] - Rows 256-511 (ROW-SHARDED)
```

**NCCL:** Broadcast from root + local extraction

---

#### 2. Replicated → Col-sharded (0.26 ms)

**Before:**

```
GPU 0: X [512, 512] - Full matrix (REPLICATED)
GPU 1: X [512, 512] - Full matrix (REPLICATED)
```

**Operation:** `X.shard(1)` - Shard along dimension 1 (columns)

**After:**

```
GPU 0: X [512, 256] - Columns 0-255 (COL-SHARDED)
GPU 1: X [512, 256] - Columns 256-511 (COL-SHARDED)
```

**NCCL:** Broadcast from root + local extraction

---

#### 3. Row-shard → Col-shard (0.38 ms) ⚠️ SLOWEST

**Before:**

```
GPU 0: X [256, 512] - Rows 0-255 (ROW-SHARDED)
GPU 1: X [256, 512] - Rows 256-511 (ROW-SHARDED)
```

**Operations:**

```cpp
X.replicate(0);  // Step 1: AllGather
X.shard(1);      // Step 2: Shard columns
```

**After Step 1 (replicate):**

```
GPU 0: X [512, 512] - Full matrix (REPLICATED)
GPU 1: X [512, 512] - Full matrix (REPLICATED)  
```

**After Step 2 (shard columns):**

```
GPU 0: X [512, 256] - Columns 0-255 (COL-SHARDED)
GPU 1: X [512, 256] - Columns 256-511 (COL-SHARDED)
```

**NCCL:** AllGather (expensive!) + local extraction

**Why slowest?** Two-stage operation: gather full tensor, then re-shard

---

#### 4. Sharded → Replicated (0.25 ms) ✅ FASTEST

**Before:**

```
GPU 0: X [512, 256] - Columns 0-255 (COL-SHARDED)
GPU 1: X [512, 256] - Columns 256-511 (COL-SHARDED)
```

**Operation:** `X.replicate(0)` - AllGather to replicate

**After:**

```
GPU 0: X [512, 512] - Full matrix (REPLICATED)
GPU 1: X [512, 512] - Full matrix (REPLICATED)
```

**NCCL:** AllGather (each GPU sends its shard to all others)

---

**Summary Table:**

| Transformation | Input State | Operation | Output State | Steps | NCCL Ops |
|----------------|-------------|-----------|--------------|-------|----------|
| Replicated → Row-shard | Full on both | `shard(0)` | Rows split | 1 | Broadcast + extract |
| Replicated → Col-shard | Full on both | `shard(1)` | Cols split | 1 | Broadcast + extract |
| Row-shard → Col-shard | Rows split | `replicate()` + `shard(1)` | Cols split | **2** | AllGather + extract |
| Sharded → Replicated | Cols split | `replicate()` | Full on both | 1 | AllGather |

---

### Key Findings

**1. Excellent Bandwidth: 4-5 GB/s for Large Tensors**

All transformations achieve good GPU communication efficiency. For 8K×8K matrices, we're hitting 5+ GB/s.

**2. Row→Column Redistribution is Slowest**

The Row→Col transformation is **1.6x slower** than simple transformations:

- Requires two operations: AllGather + local extraction  
- More complex memory access pattern

**3. Simple Transformations are Fast**

Direct replicate↔shard operations are fastest:

- Single NCCL collective (Broadcast or AllGather)
- Direct GPU-to-GPU transfer

**4. Linear Scaling**

Performance scales linearly with data size:

- 512×512 (1 MB): ~0.26 ms
- 4K×4K (64 MB): ~12 ms  
- 8K×8K (256 MB): ~49 ms

---

## Overall Takeaways

1. **Column-parallel** is communication-free - each GPU independently computes its output columns
2. **Row-parallel** needs AllReduce to sum partial results across GPUs
3. The `Layout` system tracks whether data is `REPLICATED` or `SHARDED`
4. Layout transformations are **production-ready** with 5+ GB/s bandwidth
5. Avoid cross-dimension redistribution (Row→Col) when possible - it's 1.6x slower
6. This pattern (from Megatron-LM) minimizes communication while maximizing parallelism!

**For Production Transformer Models:**

- Column-parallel linear: Use Replicated→Col-shard (fast, no sync needed)
- Row-parallel linear: Use Row-shard inputs with AllReduce at end
- Minimize layout changes between layers for best performance
