# NCCL Collectives Benchmark Tool

This tool measures communication overhead for different NCCL collective operations with varying data sizes.

## Purpose

After implementing DeviceMesh, you want to understand the performance characteristics of:
- AllReduce
- AllGather
- ReduceScatter  
- Broadcast

With different tensor sizes to quantify communication costs.

## Usage

```bash
# For 2 GPUs (as you have)
mpirun -np 2 ./nccl_benchmark
```

## Metrics Measured

For each collective operation and data size:
- **Latency**: Time taken for the operation (ms)
- **Bandwidth**: Effective data transfer rate (GB/s)
- **Algorithm bandwidth**: Theoretical bandwidth based on NCCL's algorithm

## Data Sizes

Tests run on logarithmically spaced sizes from 1KB to 1GB:
- 1 KB, 4 KB, 16 KB, 64 KB, 256 KB
- 1 MB, 4 MB, 16 MB, 64 MB, 256 MB
- 1 GB

## Output Format

```
========================================
NCCL Collectives Benchmark (2 GPUs)
========================================

--- AllReduce ---
Size: 1.00 KB | Latency: 0.05 ms | Bandwidth: 20.00 GB/s
Size: 4.00 KB | Latency: 0.06 ms | Bandwidth: 66.67 GB/s
...

--- AllGather ---
...
```
