#include "mycublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

// Helper to check CUDA/cuBLAS errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

template<typename T>
void fill_random(T* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX);
    }
}

// Specialization for half/bfloat16 if needed
template<>
void fill_random(__half* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
}

template<>
void fill_random(__nv_bfloat16* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
    }
}

// Wrapper for cuBLAS BF16 GEMM
cublasStatus_t cublasBgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const __nv_bfloat16 *alpha,
    const __nv_bfloat16 *A, int lda, long long int strideA,
    const __nv_bfloat16 *B, int ldb, long long int strideB,
    const __nv_bfloat16 *beta,
    __nv_bfloat16 *C, int ldc, long long int strideC,
    int batchCount)
{
    float alpha_f = __bfloat162float(*alpha);
    float beta_f = __bfloat162float(*beta);
    
    return cublasGemmStridedBatchedEx(
        handle, transa, transb,
        m, n, k,
        &alpha_f,
        A, CUDA_R_16BF, lda, strideA,
        B, CUDA_R_16BF, ldb, strideB,
        &beta_f,
        C, CUDA_R_16BF, ldc, strideC,
        batchCount,
        CUBLAS_COMPUTE_32F, 
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

// Benchmark Function
template<typename T, typename FuncMyBlas, typename FuncCuBlas>
void run_benchmark(
    const std::string& name,
    int M, int N, int K, int batchCount,
    FuncMyBlas myblas_func,
    FuncCuBlas cublas_func,
    T alpha, T beta,
    size_t dtype_size
) {
    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;
    
    size_t sizeA = batchCount * strideA;
    size_t sizeB = batchCount * strideB;
    size_t sizeC = batchCount * strideC;

    T *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA * dtype_size));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB * dtype_size));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC * dtype_size));

    // Fill data (device-side fill would be faster but copying valid data is safer for NaN checks if we care, here just timing)
    // For benchmark speed, just memset or leave uninit? 
    // Best to have valid data to avoid denorm slowdowns etc.
    // Let's alloc on host, fill, copy.
    T *h_A = new T[sizeA];
    T *h_B = new T[sizeB];
    fill_random(h_A, sizeA);
    fill_random(h_B, sizeB);
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA * dtype_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB * dtype_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC * dtype_size));

    mycublasHandle_t myHandle;
    mycublasCreate(&myHandle);
    
    cublasHandle_t cuHandle;
    cublasCreate(&cuHandle);
    
    // Create Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 20;
    
    // --- Verification Steps ---
    std::cout << "Verifying " << name << " output vs cuBLAS... ";
    
    // 1. Run MyBlas
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC * dtype_size)); // Reset C
    myblas_func(myHandle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C, N, strideC, batchCount);
    cudaDeviceSynchronize();
    
    // Copy MyBlas result to host
    T* h_C_myblas = new T[sizeC];
    CHECK_CUDA(cudaMemcpy(h_C_myblas, d_C, sizeC * dtype_size, cudaMemcpyDeviceToHost));
    
    // 2. Run cuBLAS
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC * dtype_size)); // Reset C
    // Note: Swapping A/B and M/N for Row-Major support in cuBLAS
    cublas_func(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, strideB, d_A, K, strideA, &beta, d_C, N, strideC, batchCount);
    cudaDeviceSynchronize();
    
    // Copy cuBLAS result to host
    T* h_C_cublas = new T[sizeC];
    CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C, sizeC * dtype_size, cudaMemcpyDeviceToHost));
    
    // 3. Compare (Relative Error)
    double max_rel_diff = 0.0;
    for(size_t i=0; i<sizeC; ++i) {
        double v1 = static_cast<double>(static_cast<float>(h_C_myblas[i]));
        double v2 = static_cast<double>(static_cast<float>(h_C_cublas[i]));
        double diff = std::abs(v1 - v2);
        double mag = std::max(std::abs(v1), std::abs(v2));
        double rel_diff = (mag > 1e-5) ? diff / mag : diff; // Use absolute if close to zero
        if(rel_diff > max_rel_diff) max_rel_diff = rel_diff;
    }
    
    // Free host comparison buffers
    delete[] h_C_myblas;
    delete[] h_C_cublas;

    std::cout << "Rel Error: " << std::scientific << max_rel_diff;
    // Tolerance: FP32 (TF32) ~1e-3, FP16/BF16 ~1e-2, FP64 ~1e-14
    double tol = (dtype_size == 4) ? 1e-3 : (dtype_size == 2 ? 2e-2 : 1e-14);
    if (max_rel_diff < tol) std::cout << " [MATCH] ";
    else std::cout << " [MISMATCH] ";
    std::cout << std::defaultfloat << std::endl;

    // --- Performance Benchmark ---
    // Warmup MyBlas
    myblas_func(myHandle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C, N, strideC, batchCount);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for(int i=0; i<iterations; ++i) {
        myblas_func(myHandle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C, N, strideC, batchCount);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_myblas = 0;
    cudaEventElapsedTime(&ms_myblas, start, stop);
    float avg_ms_myblas = ms_myblas / iterations;

    // --- cuBLAS Benchmark ---
    // Warmup cuBLAS
    cublas_func(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, strideB, d_A, K, strideA, &beta, d_C, N, strideC, batchCount);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for(int i=0; i<iterations; ++i) {
        cublas_func(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, strideB, d_A, K, strideA, &beta, d_C, N, strideC, batchCount);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_cublas = 0;
    cudaEventElapsedTime(&ms_cublas, start, stop);
    float avg_ms_cublas = ms_cublas / iterations;

    // Report
    double flops = 2.0 * static_cast<double>(M) * N * K * batchCount;
    double tflops_myblas = (flops / 1e12) / (avg_ms_myblas / 1000.0);
    double tflops_cublas = (flops / 1e12) / (avg_ms_cublas / 1000.0);

    std::cout << "| " << std::setw(6) << name 
              << " | " << std::setw(15) << (std::to_string(M)+"x"+std::to_string(N)+"x"+std::to_string(K)) 
              << " | " << std::setw(6) << batchCount
              << " | " << std::setw(10) << std::fixed << std::setprecision(3) << tflops_myblas 
              << " | " << std::setw(10) << tflops_cublas 
              << " | " << std::setw(8) << (tflops_myblas / tflops_cublas * 100.0) << "% |" << std::endl;

    delete[] h_A; delete[] h_B;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    mycublasDestroy(myHandle);
    cublasDestroy(cuHandle);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    std::cout << "Benchmarking Batched GEMM Performance (MyBlas vs cuBLAS)" << std::endl;
    std::cout << "GPU: " << "RTX 30xx/40xx (Ampere/Ada)" << std::endl;
    std::cout << std::endl;
    std::cout << "| Type   | Size (MxNxK)    | Batch  | MyBlas (TF) | cuBLAS (TF) | Rel. Perf |" << std::endl;
    std::cout << "|--------|-----------------|--------|-------------|-------------|-----------|" << std::endl;

    // SGEMM (FP32)
    run_benchmark<float>("SGEMM", 256, 256, 256, 128, mycublasSgemmStridedBatched, cublasSgemmStridedBatched, 1.0f, 0.0f, sizeof(float));
    run_benchmark<float>("SGEMM", 512, 512, 512, 64, mycublasSgemmStridedBatched, cublasSgemmStridedBatched, 1.0f, 0.0f, sizeof(float));
    
    // HGEMM (FP16)
    run_benchmark<__half>("HGEMM", 256, 256, 256, 128, mycublasHgemmStridedBatched, cublasHgemmStridedBatched, __float2half(1.0f), __float2half(0.0f), sizeof(__half));
    run_benchmark<__half>("HGEMM", 512, 512, 512, 64, mycublasHgemmStridedBatched, cublasHgemmStridedBatched, __float2half(1.0f), __float2half(0.0f), sizeof(__half));

    // BGEMM (BF16)
    run_benchmark<__nv_bfloat16>("BGEMM", 256, 256, 256, 128, mycublasBgemmStridedBatched, cublasBgemmStridedBatched, __float2bfloat16(1.0f), __float2bfloat16(0.0f), sizeof(__nv_bfloat16));
    run_benchmark<__nv_bfloat16>("BGEMM", 512, 512, 512, 64, mycublasBgemmStridedBatched, cublasBgemmStridedBatched, __float2bfloat16(1.0f), __float2bfloat16(0.0f), sizeof(__nv_bfloat16));

    // DGEMM (FP64)
    run_benchmark<double>("DGEMM", 128, 128, 128, 128, mycublasDgemmStridedBatched, cublasDgemmStridedBatched, 1.0, 0.0, sizeof(double));
    
    return 0;
}
