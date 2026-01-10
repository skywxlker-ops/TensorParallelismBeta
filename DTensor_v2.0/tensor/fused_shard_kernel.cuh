#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Launch kernel for dim 1 sharding: [D0, D1, D2] -> [D0, D1_local, D2]
void launch_shard_dim1_kernel(
    float* d_src,           // Source data (full tensor)
    float* d_dst,           // Destination (sharded tensor)
    int64_t D0, int64_t D1, int64_t D2,
    int64_t D1_local,       // D1 / world_size
    int rank,
    int64_t total_elements,
    cudaStream_t stream
);

// Launch kernel for dim 2 sharding: [D0, D1, D2] -> [D0, D1, D2_local]
void launch_shard_dim2_kernel(
    float* d_src,           // Source data (full tensor)
    float* d_dst,           // Destination (sharded tensor)
    int64_t D0, int64_t D1, int64_t D2,
    int64_t D2_local,       // D2 / world_size
    int rank,
    int64_t total_elements,
    cudaStream_t stream
);
