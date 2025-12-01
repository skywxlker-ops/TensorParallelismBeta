# Learning Report - Dec 1, 2025

## What I Built

Added GPU-native data initialization to DTensor. Now tensors load once on GPU0 and scatter via NCCL instead of copying from CPU to each GPU separately.

## The Problem

Before: Each GPU got its data chunk from CPU separately

```
CPU Data
  ├─> cudaMemcpy ──> GPU0 (shard 0)
  ├─> cudaMemcpy ──> GPU1 (shard 1)
  └─> cudaMemcpy ──> GPU2 (shard 2)
```

After: Single GPU load + GPU-to-GPU broadcast

```
CPU Data ──> GPU0 (full tensor)
              │
              └─> ncclBroadcast ──> All GPUs
                                     │
                                     └─> Local extraction
```

## Changes Made

### dtensor.h

Added new method:

```cpp
void setDataFromRoot(const std::vector<float>& host_data, 
                     const Layout& layout, int root = 0);
```

### dtensor.cpp

Implemented two functions:

- `setDataFromRoot()` - loads on root, broadcasts, extracts shards
- `_extract_local_shard()` - pulls local chunk from full tensor using GPU memcpy

Key logic:

- Replicated: direct broadcast, done
- Sharded: broadcast full tensor → each GPU extracts its piece with `cudaMemcpy2D` (for columns) or regular copy (for rows)

### Build System

Had to bypass broken autograd code:

- Removed `tensor_ops_bridge.cpp` from Makefile
- Created `tensor_ops_bridge_stub.cpp` with simple operator forwarding
- Removed `-lcgad` dependency

## Test Results

```
GPU-Native Init Tests:

[Replicated]
  PASS

[Row-Sharded]
  rank 0: PASS
  rank 1: PASS

[Column-Sharded]
  rank 0: PASS
  rank 1: PASS

Done.
```

All three layouts work:

- **Replicated**: both GPUs get full [4,4] tensor
- **Row-sharded**: GPU0 gets rows 0-3, GPU1 gets rows 4-7  
- **Column-sharded**: GPU0 gets cols 0-3, GPU1 gets cols 4-7

## What I Learned

1. NCCL broadcast is cleaner than point-to-point for this use case
2. `cudaMemcpy2D` handles non-contiguous column slicing on GPU
3. Row slicing is just offset arithmetic (contiguous memory)
4. ProcessGroup doesn't have send/recv yet, so broadcast + local slice works fine

## Trade-offs

**Good:**

- Only one CPU→GPU transfer (on root)
- GPU-to-GPU is faster than PCIe
- No CPU splitting logic

**Bad:**

- Temporarily allocates full tensor on all GPUs during init
- Could be smarter with point-to-point sends (future work)

## Files Modified

- `tensor/dtensor.h` - method declaration
- `tensor/dtensor.cpp` - ~150 lines of implementation
- `tests/test_gpu_native_init.cpp` - 200 lines of tests
- `tests/Makefile` - removed autograd deps, added stub
- `bridge/tensor_ops_bridge_stub.cpp` - minimal bridge (25 lines)

## Usage

```cpp
// Root prepares full data
std::vector<float> data;
if (rank == 0) {
    data = {1,2,3,4,5,6,7,8};  // full tensor
}

// Everyone calls setDataFromRoot
Layout layout(mesh, {8}, ShardingType::SHARDED, 0);
DTensor tensor(mesh, pg);
tensor.setDataFromRoot(data, layout, 0);

// Each GPU now has its shard
```

Done. Works.
