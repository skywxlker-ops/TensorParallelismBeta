#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "mycublas.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

int main() {
    int M = 8, N = 8, K = 4; // Use exact tile size for verification
    double alpha = 1.0;
    double beta = 0.0;

    std::vector<double> h_A(M * K, 1.0);
    std::vector<double> h_B(K * N, 2.0);
    std::vector<double> h_C(M * N);
    
    // Dot prod: size K=4. result = 1.0 * 2.0 * 4 = 8.0.

    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(double), cudaMemcpyHostToDevice));

    mycublasHandle_t handle;
    mycublasCreate(&handle);

    std::cout << "Launching DGEMM (FP64 Tensor Core)..." << std::endl;
    // Note: M, N, K must align with tile sizes in this naive implementation.
    mycublasDgemm(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));

    double result = h_C[0];
    std::cout << "C[0] = " << result << std::endl;
    
    if (abs(result - 8.0) < 0.001) {
        std::cout << "TEST PASSED!" << std::endl;
    } else {
        std::cout << "TEST FAILED! Expected 8.0" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    mycublasDestroy(handle);

    return 0;
}
