#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <stdint.h>
#include "mycublas.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

int main() {
    int M = 16, N = 16, K = 16; 
    int32_t alpha = 1;
    int32_t beta = 0;

    // Use values small enough not to overflow easily in manual check, 
    // but C is int32 so plenty of room.
    // A=2 (int8), B=3 (int8).
    // Dot prod (K=16) = 16 * (2*3) = 16*6 = 96.
    
    std::vector<int8_t> h_A(M * K, 2);
    std::vector<int8_t> h_B(K * N, 3);
    std::vector<int32_t> h_C(M * N);

    int8_t *d_A, *d_B;
    int32_t *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(int8_t), cudaMemcpyHostToDevice));

    mycublasHandle_t handle;
    mycublasCreate(&handle);

    std::cout << "Launching IGEMM (Int8 Tensor Core)..." << std::endl;
    mycublasIgemm(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int32_t result = h_C[0];
    std::cout << "C[0] = " << result << std::endl;
    
    if (result == 96) {
        std::cout << "TEST PASSED!" << std::endl;
    } else {
        std::cout << "TEST FAILED! Expected 96" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    mycublasDestroy(handle);

    return 0;
}
