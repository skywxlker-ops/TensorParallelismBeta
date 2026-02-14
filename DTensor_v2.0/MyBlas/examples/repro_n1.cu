#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "mycublas.h"

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA API failed at line %d with error: %s (%d)\n", \
               __LINE__, cudaGetErrorString(status), status); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    // Parameters matching the crash site
    // Matmul Debug: M=1024, N=1, K=768, Batch=4
    // LDA=768, LDB=1, LDC=1
    // StrideA=786432, StrideB=0, StrideC=1024
    
    int M = 1024;
    int N = 1;
    int K = 768;
    int batchCount = 4;
    
    long long strideA = (long long)M * K;
    long long strideB = 0;
    long long strideC = (long long)M * N;
    
    int lda = K;
    int ldb = N;
    int ldc = N;
    
    printf("Testing N=1 Broadcast Case:\n");
    printf("M=%d, N=%d, K=%d, Batch=%d\n", M, N, K, batchCount);
    printf("LDA=%d, LDB=%d, LDC=%d\n", lda, ldb, ldc);
    printf("StrideA=%lld, StrideB=%lld, StrideC=%lld\n", strideA, strideB, strideC);

    size_t size_A = batchCount * strideA;
    size_t size_B = K * N; // StrideB is 0, so only one matrix
    size_t size_C = batchCount * strideC;
    
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    h_A = (float*)malloc(size_A * sizeof(float));
    h_B = (float*)malloc(size_B * sizeof(float));
    h_C = (float*)malloc(size_C * sizeof(float));
    
    // Initialize
    for (size_t i = 0; i < size_A; i++) h_A[i] = 1.0f;
    for (size_t i = 0; i < size_B; i++) h_B[i] = 1.0f; // B is all ones
    // Result should be K * 1.0 = 768.0 for all elements.
    
    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));
    
    mycublasHandle_t handle;
    if (mycublasCreate(&handle) != MYCUBLAS_STATUS_SUCCESS) {
        printf("Failed to create handle\n");
        return 1;
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    mycublasSgemmStridedBatched(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, batchCount);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify
    bool correct = true;
    for (int b = 0; b < batchCount; b++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float val = h_C[b * strideC + m * ldc + n];
                if (fabs(val - 768.0f) > 1e-3) {
                    if (correct) { // Print first failure
                        printf("Mismatch at B=%d, M=%d: Expected 768.0, Got %f\n", b, m, val);
                    }
                    correct = false;
                }
            }
        }
    }
    
    if (correct) printf("Verification passed!\n");
    else printf("Verification FAILED!\n");

    mycublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return correct ? 0 : 1;
}
