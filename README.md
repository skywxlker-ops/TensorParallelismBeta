
# DTensor PyBind11 Wrapper + Tensor-Parallel MLP (Skeleton)

This is a *scaffold* that wraps your custom C++ DTensor backend (MPI+NCCL) using pybind11,
and provides a minimal Python-side MLP built with Column/Row parallel linear layers.

## Layout

- `bindings/dtensor_pybind.cpp` — pybind11 module exposing `ProcessGroup`, `DTensor`, placements, and collectives.
- `include/` — placeholder headers mimicking expected backend APIs; replace with your real headers or symlink them.
- `python/dtensor/__init__.py` — high-level Python helpers: `ColumnParallelLinear`, `RowParallelLinear`, `TinyTPMLP`.
- `examples/tp_mlp.py` — example runnable forward pass.
- `tests/test_process_group.py` — very small sanity test of all-reduce.

## Build

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
cmake --install .
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
```

You may need to specify your NCCL and backend header/library paths, e.g.:

```
cmake -DNCCL_INCLUDE_DIR=/usr/local/cuda/include \
      -DNCCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libnccl.so \
      -DDTENSOR_BACKEND_INCLUDE=/path/to/your/headers \
      -DDTENSOR_BACKEND_LIB=/path/to/libdtensor_backend.a \
      ..
```

## Run

Use `mpirun` or your launcher to spawn N ranks. Example with pure MPI and `tp-size=2`:

```
mpirun -np 4 python examples/tp_mlp.py \
  --tp-size 2 --world-size 4 --rank ${OMPI_COMM_WORLD_RANK} --backend mpi
```

On rank 0 you should see an output shape and checksum.

> NOTE: The `bindings/` currently include *placeholder* includes and function signatures.
> Replace these with your actual backend headers, types, and functions.
> The Python MLP currently stages data to host for matmul to keep the example self-contained.
> Swap those parts with device GEMMs in your backend for performance.
