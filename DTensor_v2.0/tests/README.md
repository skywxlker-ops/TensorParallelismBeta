# DTensor Tests

This directory contains all test files for the DTensor framework.

## Test Files

- **test_stream_pool.cpp** - Tests for async stream management (StreamPool, Work callbacks, chaining)
- **test_process_group.cpp** - Tests for NCCL ProcessGroup operations (collectives, send/recv, barrier)
- **test_mlp_forward.cpp** - End-to-end MLP forward pass with tensor parallelism
- **test_redistribute.cpp** - Tests for tensor redistribution across different sharding layouts

## Building Tests

Build all tests:

```bash
make all
```

Build a specific test:

```bash
make test_stream_pool
make test_process_group
make test_mlp_forward
make test_redistribute
```

## Running Tests

Run all tests:

```bash
make run_all
```

Run individual tests:

```bash
make run_stream      # StreamPool tests (single process)
make run_pg          # ProcessGroup tests (2 MPI ranks)
make run_mlp         # MLP forward pass (2 MPI ranks)
make run_redist      # Redistribute tests (2 MPI ranks)
```

## Help

View all available targets:

```bash
make help
```

## Cleanup

Remove test executables:

```bash
make clean
```

Remove test executables and core objects:

```bash
make clean_all
```
