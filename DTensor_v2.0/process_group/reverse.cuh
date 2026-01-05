#pragma once
#include <cuda_runtime.h>

// Standard C++ wrapper that g++ can understand
void launch_reverse_kernel(float* d_src, float* d_dst, int nx, int ny, int nz, int dim, cudaStream_t stream);

// Hide the actual kernel from the C++ compiler
#ifdef __CUDACC__
__global__ void reverse_kernel(float* src, float* dst, int nx, int ny, int nz, int dim);
#endif 