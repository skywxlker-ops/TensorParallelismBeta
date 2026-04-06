
#include<iostream>
#include<cuda_runtime.h>
#include "reverse.cuh"

__global__ void reverse_kernel(float* src, float* dst, 
                                int nx, int ny, int nz, 
                                int dim) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        
        int rx = x, ry = y, rz = z; 

        if (dim == 0)      rx = (nx - 1 - x);
        else if (dim == 1) ry = (ny - 1 - y);
        else               rz = (nz - 1 - z);

        int64_t srcIdx = (int64_t)x * ny * nz + (int64_t)y * nz + z;
        int64_t dstIdx = (int64_t)rx * ny * nz + (int64_t)ry * nz + rz;

        dst[dstIdx] = src[srcIdx];
    }
}


void launch_reverse_kernel(float* d_src, float* d_dst, int nx, int ny, int nz, int dim, cudaStream_t stream) {
    dim3 threads(8, 8, 8);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    reverse_kernel<<<blocks, threads, 0, stream>>>(d_src, d_dst, nx, ny, nz, dim);
}

