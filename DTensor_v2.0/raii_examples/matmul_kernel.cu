#include <cuda_runtime.h>

// ============================================================================
// Simple GPU Matrix Multiplication Kernel
// ============================================================================

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                             int M, int K, int N) {
    // C = A @ B
    // A: [M, K], B: [K, N], C: [M, N]
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host wrapper function
void launch_matmul_kernel(const float* d_A, const float* d_B, float* d_C,
                         int M, int K, int N) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    
    // Check for kernel launch errors (caller will handle errors)
    cudaGetLastError();
}
