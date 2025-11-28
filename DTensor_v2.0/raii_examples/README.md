# RAII Examples

This directory contains educational examples demonstrating **RAII (Resource Acquisition Is Initialization)** in C++.

## What is RAII?

RAII is a C++ programming idiom where:

- **Constructor acquires resources** (memory, file handles, locks, etc.)
- **Destructor releases resources** automatically when object goes out of scope
- **Resource lifetime = Object lifetime**

## Examples Included

### 1. Simple Array (`01_simple_array.cpp`)

- Basic demonstration of RAII with dynamic arrays
- Compares manual `new`/`delete` vs `std::vector`

### 2. CPU Matrix Multiplication (`02_cpu_matmul.cpp`)

- Matrix multiplication with multiple allocations
- Shows RAII benefits with multiple resources

### 3. GPU Matrix Multiplication (`03_gpu_matmul.cpp`)

- GPU memory management with CUDA
- Demonstrates custom RAII wrappers for GPU resources

### 4. **Benchmark** (`04_benchmark.cpp`) 🔥

- **Performance comparison**: Timing allocation/deallocation
- **Exception safety**: Demonstrates automatic cleanup
- **Code complexity**: Shows cleanup code reduction
- **Memory leak resistance**: Simulates leak scenarios

## Building

```bash
# Compile all examples
make all

# Compile individual examples
make simple_array
make benchmark      # ⭐ Performance & safety comparison
make cpu_matmul
make gpu_matmul

# Run examples
./bin/01_simple_array
./bin/02_cpu_matmul
./bin/03_gpu_matmul
./bin/04_benchmark  # ⭐ See the difference!

# Clean
make clean
```

## Key Takeaways

| Without RAII | With RAII |
|--------------|-----------|
| Manual cleanup in every exit path | Automatic cleanup |
| Exception-unsafe | Exception-safe |
| Error-prone | Compiler-enforced safety |
| Memory leaks possible | Guaranteed cleanup |

## RAII Guarantee

**If an object is successfully constructed, its destructor WILL be called when it goes out of scope, even if exceptions are thrown.**

This is the foundation of exception-safe, leak-free C++ code.
