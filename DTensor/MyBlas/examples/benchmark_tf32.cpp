#include "mycublas.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iomanip>

// cuBLAS handle (thread-local for safety)
cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

void benchmark_cublas_sgemm(int M, int N, int K, bool use_tf32, int iterations = 100) {
    // Allocate matrices
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    
    // Initialize with random data
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Get cuBLAS handle
    cublasHandle_t handle = get_cublas_handle();
    
    // Set math mode
    if (use_tf32) {
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    } else {
        cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);  // Pure FP32, no Tensor Cores
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time_ms = milliseconds / iterations;
    
    // Calculate TFLOPS
    double flops = 2.0 * (double)M * N * K;
    double tflops = (flops / 1e12) / (avg_time_ms / 1000.0);
    double gflops = tflops * 1000.0;
    
    // Calculate bandwidth
    double bytes = (double)(M * K + K * N + M * N) * sizeof(float);
    double bandwidth_gb_s = (bytes / 1e9) / (avg_time_ms / 1000.0);
    
    std::cout << "\n=== " << (use_tf32 ? "TF32 Tensor Cores" : "FP32 CUDA Cores") << " ===" << std::endl;
    std::cout << "Matrix Size: " << M << "x" << K << " x " << K << "x" << N << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // Peak TFLOPS
    double peak_cuda_cores = 12.74;  // RTX 3060
    double peak_tf32 = 25.0;         // RTX 3060 TF32
    double peak = use_tf32 ? peak_tf32 : peak_cuda_cores;
    std::cout << "Efficiency: " << (tflops / peak * 100.0) << "% of peak (" << peak << " TFLOPS)" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "cuBLAS FP32 vs TF32 Comparison" << std::endl;
    std::cout << "RTX 3060: 12.74 TFLOPS (FP32), 25 TFLOPS (TF32)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test various sizes
    int sizes[] = {512, 1024, 2048, 4096, 8192};
    
    for (int size : sizes) {
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Matrix Size: " << size << "x" << size << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // FP32 CUDA Cores (no Tensor Cores)
        int iters = (size <= 1024) ? 1000 : (size <= 2048) ? 100 : (size <= 4096) ? 50 : 10;
        benchmark_cublas_sgemm(size, size, size, false, iters);
        
        // TF32 Tensor Cores
        benchmark_cublas_sgemm(size, size, size, true, iters);
        
        // Calculate speedup
        // (We'll do this manually by comparing the outputs)
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
