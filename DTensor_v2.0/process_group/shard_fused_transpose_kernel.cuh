#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Fused transpose + contiguous kernel
// Reads from [A, B, C] layout with strides, writes to contiguous [C, B, A] layout
void launch_fused_transpose_contiguous_kernel(
    float* d_src,
    float* d_dst,
    int64_t dim0, int64_t dim1, int64_t dim2,  // Source dimensions
    int64_t s0, int64_t s1, int64_t s2,         // Source strides
    int swap_dim0, int swap_dim1,               // Which dims to swap (e.g., 0,2 for transpose(0,2))
    int64_t total_elements,
    cudaStream_t stream
);
