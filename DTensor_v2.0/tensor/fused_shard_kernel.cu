#include <iostream>
#include <cuda_runtime.h>
#include "fused_shard_kernel.cuh"

// Kernel for dim 1 sharding: [D0, D1, D2] -> [D0, D1_local, D2]
__global__ void shard_dim1_kernel(
    float* src,
    float* dst,
    int64_t D0, int64_t D1, int64_t D2,
    int64_t D1_local,
    int rank,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Destination coordinates [d0, d1_local, d2] in output [D0, D1_local, D2]
        int64_t d2 = idx % D2;
        int64_t temp = idx / D2;
        int64_t d1_local = temp % D1_local;
        int64_t d0 = temp / D1_local;
        
        // Global d1 coordinate
        int64_t d1_global = rank * D1_local + d1_local;
        
        // Source index in [D0, D1, D2] layout
        int64_t src_idx = d0 * (D1 * D2) + d1_global * D2 + d2;
        
        dst[idx] = src[src_idx];
    }
}

// Kernel for dim 2 sharding: [D0, D1, D2] -> [D0, D1, D2_local]
__global__ void shard_dim2_kernel(
    float* src,
    float* dst,
    int64_t D0, int64_t D1, int64_t D2,
    int64_t D2_local,
    int rank,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Destination coordinates [d0, d1, d2_local] in output [D0, D1, D2_local]
        int64_t d2_local = idx % D2_local;
        int64_t temp = idx / D2_local;
        int64_t d1 = temp % D1;
        int64_t d0 = temp / D1;
        
        // Global d2 coordinate
        int64_t d2_global = rank * D2_local + d2_local;
        
        // Source index in [D0, D1, D2] layout
        int64_t src_idx = d0 * (D1 * D2) + d1 * D2 + d2_global;
        
        dst[idx] = src[src_idx];
    }
}

void launch_shard_dim1_kernel(
    float* d_src,
    float* d_dst,
    int64_t D0, int64_t D1, int64_t D2,
    int64_t D1_local,
    int rank,
    int64_t total_elements,
    cudaStream_t stream
) {
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    
    shard_dim1_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_src, d_dst, D0, D1, D2, D1_local, rank, total_elements
    );
}

void launch_shard_dim2_kernel(
    float* d_src,
    float* d_dst,
    int64_t D0, int64_t D1, int64_t D2,
    int64_t D2_local,
    int rank,
    int64_t total_elements,
    cudaStream_t stream
) {
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    
    shard_dim2_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_src, d_dst, D0, D1, D2, D2_local, rank, total_elements
    );
}
