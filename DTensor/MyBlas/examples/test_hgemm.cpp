#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "mycublas.h"

// Helper to check CUDA errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

int main() {
    int M = 16, N = 16, K = 16;
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    // Host inputs (float for easy init)
    // Using 16x16 to match one Tensor Core tile exactly
    std::vector<float> h_A_float(M * K, 1.0f);
    std::vector<float> h_B_float(K * N, 2.0f); // All 2s
    
    // C = A * B. Row i of A dot Col j of B.
    // Row of A is all 1s (size K=16). Col of B is all 2s (size K=16).
    // Dot prod = 16 * (1*2) = 32.
    
    // Allocate Host Half
    std::vector<__half> h_A(M * K);
    std::vector<__half> h_B(K * N);
    std::vector<__half> h_C(M * N);
    
    for(int i=0; i<M*K; i++) h_A[i] = __float2half(h_A_float[i]);
    for(int i=0; i<K*N; i++) h_B[i] = __float2half(h_B_float[i]);

    // Device Alloc
    __half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

    // Create Handle
    mycublasHandle_t handle;
    mycublasCreate(&handle);

    std::cout << "Launching HGEMM (Tensor Core)..." << std::endl;
    mycublasHgemm(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost));

    // Verify
    float result = __half2float(h_C[0]);
    std::cout << "C[0] = " << result << std::endl;
    
    if (abs(result - 32.0f) < 0.1f) {
        std::cout << "TEST PASSED!" << std::endl;
    } else {
        std::cout << "TEST FAILED! Expected 32.0" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    mycublasDestroy(handle);

    return 0;
}
