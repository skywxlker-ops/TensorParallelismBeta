#include <cuda_runtime.h>
#include "shard_fused_transpose_kernel.cuh"

// Kernel that reads from strided source with transpose and writes contiguous
// For transpose(0, 2): source [A, B, C] -> destination [C, B, A]
__global__ void fused_transpose_contiguous_kernel(
    float* src,
    float* dst,
    int64_t dim0, int64_t dim1, int64_t dim2,
    int64_t s0, int64_t s1, int64_t s2,
    int swap_dim0, int swap_dim1,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // For transpose(0, 2): output is [dim2, dim1, dim0]
        // idx is linear index in destination (contiguous)
        
        int64_t d0, d1, d2;  // Destination coordinates
        
        if (swap_dim0 == 0 && swap_dim1 == 2) {
            // transpose(0, 2): dst shape is [dim2, dim1, dim0]
            int64_t temp = idx;
            d2 = temp % dim0;  // fastest changing in dst = dim0 of src
            temp /= dim0;
            d1 = temp % dim1;
            d0 = temp / dim1;  // slowest changing in dst = dim2 of src
            
            // Source coordinates: dst[z, y, x] <- src[x, y, z]
            int64_t src_idx = d2 * s0 + d1 * s1 + d0 * s2;
            dst[idx] = src[src_idx];
        }
        else if (swap_dim0 == 0 && swap_dim1 == 1) {
            // transpose(0, 1): dst shape is [dim1, dim0, dim2]
            int64_t temp = idx;
            d2 = temp % dim2;
            temp /= dim2;
            d1 = temp % dim0;  
            d0 = temp / dim0;
            
            // dst[y, x, z] <- src[x, y, z]
            int64_t src_idx = d1 * s0 + d0 * s1 + d2 * s2;
            dst[idx] = src[src_idx];
        }
        else if (swap_dim0 == 1 && swap_dim1 == 2) {
            // transpose(1, 2): dst shape is [dim0, dim2, dim1]
            int64_t temp = idx;
            d2 = temp % dim1;
            temp /= dim1;
            d1 = temp % dim2;
            d0 = temp / dim2;
            
            // dst[x, z, y] <- src[x, y, z]
            int64_t src_idx = d0 * s0 + d2 * s1 + d1 * s2;
            dst[idx] = src[src_idx];
        }
    }
}

void launch_fused_transpose_contiguous_kernel(
    float* d_src,
    float* d_dst,
    int64_t dim0, int64_t dim1, int64_t dim2,
    int64_t s0, int64_t s1, int64_t s2,
    int swap_dim0, int swap_dim1,
    int64_t total_elements,
    cudaStream_t stream
) {
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    
    fused_transpose_contiguous_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_src, d_dst, dim0, dim1, dim2, s0, s1, s2, swap_dim0, swap_dim1, total_elements
    );
}
