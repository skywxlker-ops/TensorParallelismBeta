#pragma once
#include <cuda_runtime.h>
#include <cstdint>


void launch_fused_rotate_kernel(
    float* d_src, 
    float* d_dst, 
    int64_t s0, int64_t s1, int64_t s2, 
    int ndim,
    int64_t total_elements,
    int64_t nx, int64_t ny, int64_t nz, 
    int axis_to_reverse,
    cudaStream_t stream
);
