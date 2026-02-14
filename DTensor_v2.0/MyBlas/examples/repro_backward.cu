#include <iostream>
#include <vector>
#include "mycublas.h"

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA API failed at line %d with error: %s (%d)\n", \
               __LINE__, cudaGetErrorString(status), status); \
        exit(EXIT_FAILURE); \
    } \
}

void run_gemm(mycublasHandle_t handle, int M, int N, int K, int batchCount, const char* name) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Standard strides for contiguous matrices
    long long strideA = (long long)M * K;
    long long strideB = (long long)K * N;
    long long strideC = (long long)M * N;
    
    size_t size_A = (size_t)batchCount * strideA * sizeof(float);
    size_t size_B = (size_t)batchCount * strideB * sizeof(float);
    size_t size_C = (size_t)batchCount * strideC * sizeof(float);

    printf("Testing %s: M=%d N=%d K=%d Batch=%d\n", name, M, N, K, batchCount);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    CHECK_CUDA(cudaMemset(d_A, 0, size_A));
    CHECK_CUDA(cudaMemset(d_B, 0, size_B));
    CHECK_CUDA(cudaMemset(d_C, 0, size_C));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mycublasSgemmStridedBatched(handle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C, N, strideC, batchCount);
    cudaEventRecord(stop);
    
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    printf("  PASSED\n");
}

int main() {
    mycublasHandle_t handle;
    if (mycublasCreate(&handle) != MYCUBLAS_STATUS_SUCCESS) return 1;

    // Backward Pass Logic
    // dInput = dLogits @ W.T
    // M = Batch*T = 1024
    // N = Hidden = 768
    // K = Vocab = 50304
    // Batch=1
    run_gemm(handle, 1024, 768, 50304, 1, "Backward Projection (Large K)");

    // dWeights = dLogits.T @ Input
    // M = Vocab = 50304
    // N = Hidden = 768
    // K = Batch*T = 1024
    // Batch=1
    run_gemm(handle, 50304, 768, 1024, 1, "Backward Weights (Large M)");

    mycublasDestroy(handle);
    return 0;
}
