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
    // Parameters for Final Projection
    // GenMatmul Debug: M=1024, N=50304, K=768, Batch=4, LDA=768, LDB=50304, LDC=50304
    // StrideA=786432, StrideB=0, StrideC=51511296
    
    int M = 1024;
    int N = 50304;
    int K = 768;
    int batchCount = 4;
    
    long long strideA = (long long)M * K;
    long long strideB = 0;
    long long strideC = (long long)M * N;
    
    int lda = K;
    int ldb = N;
    int ldc = N;
    
    printf("Testing Large N Broadcast Case:\n");
    printf("M=%d, N=%d, K=%d, Batch=%d\n", M, N, K, batchCount);
    printf("LDA=%d, LDB=%d, LDC=%d\n", lda, ldb, ldc);
    printf("StrideA=%lld, StrideB=%lld, StrideC=%lld\n", strideA, strideB, strideC);

    size_t size_A = batchCount * strideA;
    size_t size_B = (size_t)K * N; 
    size_t size_C = batchCount * strideC;
    
    // Allocate Host
    // Warning: Large allocation. 
    // C: 4 * 1024 * 50304 * 4 bytes = ~800 MB.
    // B: 768 * 50304 * 4 bytes = ~150 MB.
    // Total ~1GB. Fits in RAM.
    
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size_A * sizeof(float));
    h_B = (float*)malloc(size_B * sizeof(float));
    h_C = (float*)malloc(size_C * sizeof(float));
    
    // Initialize
    // A=1, B=1 -> C=768.
    // Use parallel init for speed
    #pragma omp parallel for
    for (size_t i = 0; i < size_A; i++) h_A[i] = 1.0f;
    #pragma omp parallel for
    for (size_t i = 0; i < size_B; i++) h_B[i] = 1.0f;
    
    float *d_A, *d_B, *d_C;
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
    
    printf("Launching Kernel...\n");
    mycublasSgemmStridedBatched(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, batchCount);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Kernel Finished.\n");
    
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify (Check a few elements)
    bool correct = true;
    for (int b = 0; b < batchCount; b++) {
        // Check corner, middle, end
        int checkpoints[] = {0, N/2, N-1};
        for (int m = 0; m < M; m+=100) {
            for (int n : checkpoints) {
                float val = h_C[b * strideC + m * ldc + n];
                if (fabs(val - 768.0f) > 1e-3) {
                     printf("Mismatch at B=%d, M=%d, N=%d: Expected 768.0, Got %f\n", b, m, n, val);
                     correct = false;
                     break;
                }
            }
            if(!correct) break;
        }
        if(!correct) break;
    }
    
    if (correct) printf("Verification passed!\n");
    else printf("Verification FAILED!\n");

    mycublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return correct ? 0 : 1;
}
