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

void run_gemm(mycublasHandle_t handle, int M, int N, int K, int batchCount, double *total_time) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    long long strideA = (long long)M * K; // Standard
    long long strideB = (long long)K * N; // Standard
    long long strideC = (long long)M * N;
    
    // Handle broadcasting logic from GenMatmul logic
    // If N=1 (Vector), B is [K, 1]. strideB is K.
    // GPT2 specific: 
    // Projection: A [B, M, K], W [K, N] (Broadcast). strideB=0.
    // If we want to test "Broadcast B" specifically.
    // The logs showed:
    // N=50304 case: StrideB=0.
    // N=1 case: StrideB=0? No, logs said StrideB=0 for some?
    // Let's rely on the logs I saw.
    // Step 537 log for B=1:
    // M=1024, N=50304... StrideB=38633472 (Which is K*N = 768*50304). So NOT broadcast?
    // Wait. Step 507 log (B=4): StrideB=0.
    // This implies `repro_large` was right to use StrideB=0 for B=4.
    // But for B=1, StrideB doesn't matter (since batch=1).
    
    // Let's use standard strides for B=1 to be safe, or 0.
    
    size_t size_A = (size_t)batchCount * strideA * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float); // Single matrix B if broadcast
    if (batchCount > 1) strideB = 0; // Simulate broadcast for B>1
    
    size_t size_C = (size_t)batchCount * strideC * sizeof(float);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    cudaMemset(d_A, 0, size_A);
    cudaMemset(d_B, 0, size_B);
    cudaMemset(d_C, 0, size_C);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mycublasSgemmStridedBatched(handle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C, N, strideC, batchCount);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    *total_time += milliseconds;

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    // printf("Run M=%d N=%d K=%d B=%d: %.3f ms\n", M, N, K, batchCount, milliseconds);
}

int main() {
    mycublasHandle_t handle;
    if (mycublasCreate(&handle) != MYCUBLAS_STATUS_SUCCESS) return 1;
    
    int B = 1; 
    // Sequence from logs
    std::vector<std::tuple<int, int, int>> sequence;
    sequence.push_back(std::make_tuple(1024, 1, 768));
    sequence.push_back(std::make_tuple(1024, 3072, 768));
    sequence.push_back(std::make_tuple(1024, 768, 3072));
    sequence.push_back(std::make_tuple(1024, 1, 768));
    sequence.push_back(std::make_tuple(1024, 1, 768));
    sequence.push_back(std::make_tuple(1024, 3072, 768));
    sequence.push_back(std::make_tuple(1024, 768, 3072));
    sequence.push_back(std::make_tuple(1024, 1, 768));
    sequence.push_back(std::make_tuple(1024, 1, 768));
    sequence.push_back(std::make_tuple(1024, 50304, 768)); // Final projection
    
    printf("Running Sequence Test with B=%d...\n", B);
    double total_time = 0;
    
    for (int i=0; i<10; i++) { // Repeat the block a few times
        for (const auto& item : sequence) {
            int m = std::get<0>(item);
            int n = std::get<1>(item);
            int k = std::get<2>(item);
            run_gemm(handle, m, n, k, B, &total_time);
        }
    }
    
    printf("Sequence Finished Successfully. Total Kernel Time: %.3f ms\n", total_time);
    
    mycublasDestroy(handle);
    return 0;
}
