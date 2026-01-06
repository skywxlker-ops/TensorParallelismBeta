# DTensor Tests

This directory contains test files for the Unparalleled (DTensor) framework.

## Prerequisites

Before running tests, build the library first:

```bash
cd DTensor_v2.0
make lib    # Creates lib/unparalleled.a and lib/unparalleled.so
```

## Building & Running Tests

All tests link against `lib/unparalleled.a` for fast compilation.

### Quick Reference

| Test | Build Command | Run Command |
|------|---------------|-------------|
| Device Mesh | `make test_device_mesh` | `mpirun -np 2 ./tests/test_device_mesh` |
| MatMul | `make test_matmul` | `mpirun -np 2 ./tests/test_matmul` |
| DTensor Factories | `make test_dtensor_factories` | `mpirun -np 2 ./tests/test_dtensor_factories` |
| DTensor Creation | `make test_dtensor_creation_methods` | `mpirun -np 2 ./tests/test_dtensor_creation_methods` |
| MLP Benchmark | `make test_mlp_benchmark` | `mpirun -np 2 ./tests/test_mlp_benchmark` |
| Rotate3D Sharding | `make test_rotate3d_sharding` | `mpirun -np 2 ./tests/test_rotate3d_sharding` |

### Example

```bash
# From DTensor_v2.0 directory
make lib                      # Build library (if not already built)
make test_dtensor_factories   # Compile test
mpirun -np 2 ./tests/test_dtensor_factories  # Run with 2 GPUs
```

## Test Descriptions

| File | Purpose |
|------|---------|
| `test_device_mesh.cpp` | Tests DeviceMesh creation and GPU assignment |
| `test_matmul.cpp` | Tests distributed matrix multiplication |
| `test_dtensor_factories.cpp` | Tests DTensor factory functions (zeros, ones, randn) |
| `test_dtensor_creation_methods.cpp` | Tests DTensor creation and initialization |
| `test_mlp_benchmark.cpp` | Benchmarks tensor-parallel MLP performance |
| `test_rotate3d_sharding.cpp` | Tests 3D tensor rotation with sharding |
| `test_lazy_partial_matmul.cpp` | Tests lazy partial reduction in row-parallel matmul |
| `test_redistribute.cpp` | Tests tensor redistribution across layouts |

## MLP Testing Suite

Additional MLP-specific tests in `mlp_testing/`:

| File | Purpose |
|------|---------|
| `test_mlp_async_timing.cpp` | Measures async vs sync communication timing |
| `test_mlp_arithmetic_intensity.cpp` | Analyzes compute vs communication ratio |
| `test_mlp_interconnect_bandwidth.cpp` | Measures GPU interconnect bandwidth |
| `test_mlp_kernel_vs_sync.cpp` | Compares kernel execution patterns |

## Cleanup

```bash
make clean    # Removes all build artifacts including lib/objects/
```
