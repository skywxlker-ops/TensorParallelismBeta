#include <iostream>
#include <vector>
#include "mycublas.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA API failed at line %d with error: %s (%d)\n", \
               __LINE__, cudaGetErrorString(status), status); \
        exit(EXIT_FAILURE); \
    } \
}

void run_gemm_async(mycublasHandle_t handle, int M, int N, int K, int batchCount, cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    long long strideA = (long long)M * K;
    long long strideB = (long long)K * N;
    long long strideC = (long long)M * N;
    
    // N=50304 logic from GenMatmul
    // LDB = 50304. StrideB = 768 * 50304 = 38633472.
    // LDC = 50304.
    
    size_t size_A = (size_t)batchCount * strideA * sizeof(float);
    size_t size_B = (size_t)batchCount * strideB * sizeof(float);
    size_t size_C = (size_t)batchCount * strideC * sizeof(float);

    printf("Testing Async M=%d N=%d K=%d Batch=%d\n", M, N, K, batchCount);

    float *d_A, *d_B, *d_C;
    // Use Async Allocator
    CHECK_CUDA(cudaMallocAsync(&d_A, size_A, stream));
    CHECK_CUDA(cudaMallocAsync(&d_B, size_B, stream));
    CHECK_CUDA(cudaMallocAsync(&d_C, size_C, stream));
    
    CHECK_CUDA(cudaMemsetAsync(d_A, 0, size_A, stream));
    CHECK_CUDA(cudaMemsetAsync(d_B, 0, size_B, stream));
    CHECK_CUDA(cudaMemsetAsync(d_C, 0, size_C, stream));

    mycublasSetStream(handle, stream);
    mycublasSgemmStridedBatched(handle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C, N, strideC, batchCount);
    
    // Sync to force failure if OOB happened during kernel
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Free Async
    CHECK_CUDA(cudaFreeAsync(d_A, stream));
    CHECK_CUDA(cudaFreeAsync(d_B, stream));
    CHECK_CUDA(cudaFreeAsync(d_C, stream));
    
    // Final Sync to trigger Free errors
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    printf("  PASSED\n");
}

int main() {
    mycublasHandle_t handle;
    if (mycublasCreate(&handle) != MYCUBLAS_STATUS_SUCCESS) return 1;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Test the Sequence leading to crash
    // 1. Projection (Forward)
    // M=1024, N=50304, K=768
    run_gemm_async(handle, 1024, 50304, 768, 1, stream);
    
    // 2. Next Op (e.g. Bias or Backward)
    // M=1024, N=1, K=768
    run_gemm_async(handle, 1024, 1, 768, 1, stream);
    
    // 3. Repeat to stress allocator
    for(int i=0; i<5; i++) {
        run_gemm_async(handle, 1024, 50304, 768, 1, stream);
        run_gemm_async(handle, 1024, 1, 768, 1, stream);
    }

    cudaStreamDestroy(stream);
    mycublasDestroy(handle);
    return 0;
}
