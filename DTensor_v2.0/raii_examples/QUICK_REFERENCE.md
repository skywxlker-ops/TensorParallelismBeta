# RAII Examples - Quick Reference

## Files Created

```
raii_examples/
├── README.md                 # Overview and compilation instructions
├── Makefile                  # Build system for all examples
├── 01_simple_array.cpp       # ✅ TESTED: Basic RAII with std::vector
├── 02_cpu_matmul.cpp         # ✅ TESTED: CPU matmul (512×256 @ 256×1024)
├── 03_gpu_matmul.cpp         # ✅ TESTED: GPU matmul with custom wrapper
├── gpu_array.h               # Custom RAII wrapper for GPU memory
└── matmul_kernel.cu          # CUDA kernel implementation
```

## Quick Start

```bash
cd raii_examples

# Build all CPU examples (works without GPU)
make simple_array cpu_matmul

# Build GPU example (requires CUDA)
make gpu_matmul

# Run examples
./bin/01_simple_array
./bin/02_cpu_matmul
./bin/03_gpu_matmul
```

## Test Results ✅

All examples compiled and ran successfully:

- **Example 1 (Simple Array)**: Demonstrates basic RAII with `std::vector` vs manual `new`/`delete`
- **Example 2 (CPU Matmul)**: Shows RAII with 3 allocations, result verified: `C[0] = 512`
- **Example 3 (GPU Matmul)**: Custom `GPUArray` wrapper managing 6 allocations (3 CPU + 3 GPU), result verified: `C[0] = 512`

## Key Code Pattern

### Without RAII ❌

```cpp
float* data = new float[1000];
// ... use data ...
if (error) {
    delete[] data;  // Must remember!
    return;
}
delete[] data;  // Leaks if exception!
```

### With RAII ✅

```cpp
std::vector<float> data(1000);
// ... use data ...
if (error) {
    return;  // Auto cleanup!
}
// Auto cleanup!
```

## The GPUArray Wrapper

The custom `GPUArray<T>` class demonstrates creating your own RAII wrapper:

```cpp
template<typename T>
class GPUArray {
    T* ptr_;
    size_t size_;
public:
    GPUArray(size_t n) {  // Constructor acquires
        cudaMalloc(&ptr_, n * sizeof(T));
    }
    ~GPUArray() {         // Destructor releases
        if (ptr_) cudaFree(ptr_);
    }
    // ... move semantics, no copy ...
};
```

**Usage:**

```cpp
GPUArray<float> d_data(1024);  // Auto allocate
d_data.copyFrom(host_data);
// ... use d_data ...
// Auto cleanup when goes out of scope!
```

## Why RAII Matters for DTensor

These patterns directly apply to fixing memory leaks in DTensor:

1. **`redistribute()` temp buffer** (dtensor.cpp:271) → Use `GPUArray<float>`
2. **NCCL communicators** → Wrap in RAII class
3. **CUDA streams/events** → Wrap in RAII class
4. **All kernel temp allocations** → Use RAII wrappers

**Result:** Zero manual cleanup, guaranteed exception safety!
