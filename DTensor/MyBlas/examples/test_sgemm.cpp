#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "mycublas.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        return -1; \
    } \
}

int main() {
    int M = 2, N = 2, K = 2;
    float alpha = 1.0f, beta = 0.0f;

    // Host data (Row Major inputs, but BLAS assumes Col Major, be careful or transpose mentally)
    // A = [1 2]
    //     [3 4]
    // B = [1 0]
    //     [0 1]
    // C = A * B = A
    
    std::vector<float> h_A = {1, 3, 2, 4}; // Col-major: 1,3 is col 1; 2,4 is col 2 -> [[1,2],[3,4]]
    std::vector<float> h_B = {1, 0, 0, 1}; // Identity
    std::vector<float> h_C(M * N, 0);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize Library
    mycublasHandle_t handle;
    if (mycublasCreate(&handle) != MYCUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create mycublas handle" << std::endl;
        return 1;
    }

    std::cout << "Launching SGEMM..." << std::endl;
    // Call our library function
    mycublasSgemm(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);

    // Sync just to be sure (though mycublasSgemm is likely async, the copy below will sync)
    // cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Result C (Col-Major):" << std::endl;
    for (int i = 0; i < M * N; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Expected: 1 3 2 4
    if (h_C[0] == 1 && h_C[1] == 3 && h_C[2] == 2 && h_C[3] == 4) {
        std::cout << "TEST PASSED!" << std::endl;
    } else {
        std::cout << "TEST FAILED!" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    mycublasDestroy(handle);

    return 0;
}
