#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
    __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    __nv_bfloat16 beta = __float2bfloat16(0.0f);

    std::vector<float> h_A_float(M * K, 1.0f);
    std::vector<float> h_B_float(K * N, 2.0f);
    
    std::vector<__nv_bfloat16> h_A(M * K);
    std::vector<__nv_bfloat16> h_B(K * N);
    std::vector<__nv_bfloat16> h_C(M * N);
    
    for(int i=0; i<M*K; i++) h_A[i] = __float2bfloat16(h_A_float[i]);
    for(int i=0; i<K*N; i++) h_B[i] = __float2bfloat16(h_B_float[i]);

    __nv_bfloat16 *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    mycublasHandle_t handle;
    mycublasCreate(&handle);

    std::cout << "Launching BGEMM (BF16 Tensor Core)..." << std::endl;
    mycublasBgemm(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    float result = __bfloat162float(h_C[0]);
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
