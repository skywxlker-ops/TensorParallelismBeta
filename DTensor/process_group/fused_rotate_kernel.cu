#include <iostream>
#include <cuda_runtime.h>
#include "fused_rotate_kernel.cuh"

#define MAX_DIMS 8

__global__ void fused_rotate_kernel(
    float* src, 
    float* dst, 
    int64_t s0, int64_t s1, int64_t s2, // Strides
    int ndim,
    int64_t total_elements,
    int nx, int ny, int nz, // Destination dimensions (contiguous)
    int axis_to_reverse
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {

        int64_t temp_idx = idx;
        int z = temp_idx % nz;
        temp_idx /= nz;
        int y = temp_idx % ny;
        int x = temp_idx / ny;


        int rx = x, ry = y, rz = z;
        if (axis_to_reverse == 0)      rx = (nx - 1 - x);
        else if (axis_to_reverse == 1) ry = (ny - 1 - y);
        else                           rz = (nz - 1 - z);

        int64_t src_offset = 0;
        

        if (ndim == 3) {
            src_offset += rx * s0;
            src_offset += ry * s1;
            src_offset += rz * s2;
        }
        
        dst[idx] = src[src_offset];
    }
}

void launch_fused_rotate_kernel(
    float* d_src, 
    float* d_dst, 
    int64_t s0, int64_t s1, int64_t s2,
    int ndim,
    int64_t total_elements,
    int64_t nx, int64_t ny, int64_t nz,
    int axis_to_reverse,
    cudaStream_t stream
) {
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    
    fused_rotate_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_src, d_dst, s0, s1, s2, ndim, total_elements, nx, ny, nz, axis_to_reverse
    );
}
