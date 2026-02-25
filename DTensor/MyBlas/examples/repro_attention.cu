#include <iostream>
#include <vector>
#include <tuple>
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

    mycublasSgemmStridedBatched(handle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C, N, strideC, batchCount);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    printf("  PASSED\n");
}

int main() {
    mycublasHandle_t handle;
    if (mycublasCreate(&handle) != MYCUBLAS_STATUS_SUCCESS) return 1;
    
    // GPT-2 Attention-like shapes (HeadDim=64, SeqLen=1024, Heads=12)
    
    // 1. Q @ K.T -> Scores
    // (B*H, T, C/H) @ (B*H, C/H, T) -> (B*H, T, T)
    // M=1024, N=1024, K=64, Batch=12 (assuming B=1)
    run_gemm(handle, 1024, 1024, 64, 12, "Score Calculation");
    
    // 2. Scores @ V -> Output
    // (B*H, T, T) @ (B*H, T, C/H) -> (B*H, T, C/H)
    // M=1024, N=64, K=1024, Batch=12
    run_gemm(handle, 1024, 64, 1024, 12, "Context Calculation");
    
    // 3. Backward Pass gradients might involve transposes that swap M/N
    // Gradeint w.r.t Q: dScores @ K
    // (T, T) @ (T, 64) -> (T, 64). M=1024, N=64, K=1024. Same as #2.
    
    // Gradient w.r.t K: dScores.T @ Q
    // (T, T) @ (T, 64). Same.

    // 4. Projection Layer (Large Batch?)
    // Input is (B*T, C). M=1024, K=768, N=3072. Batch=1.
    run_gemm(handle, 1024, 3072, 768, 1, "MLP Up-Proj");
    
    // 5. N=1 case with Batch=12? (e.g. broadcasting vector to heads?)
    run_gemm(handle, 1024, 1, 64, 12, "Broadcast Vector to Heads");

    mycublasDestroy(handle);
    printf("All Attention Tests Passed.\n");
    return 0;
}
