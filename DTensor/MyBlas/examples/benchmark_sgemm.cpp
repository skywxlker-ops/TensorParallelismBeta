#include "mycublas.h"
#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <vector>

void benchmark_sgemm(int M, int N, int K, int iterations = 100) {
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
    
    // Initialize Library
    mycublasHandle_t handle;
    mycublasCreate(&handle);

    // Warmup
    for (int i = 0; i < 10; i++) {
        mycublasSgemm(handle, M, N, K, 1.0f, d_A, K, d_B, N, 0.0f, d_C, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        mycublasSgemm(handle, M, N, K, 1.0f, d_A, K, d_B, N, 0.0f, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    mycublasDestroy(handle);
    
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
    
    std::cout << "\n=== Matrix Size: " << M << "x" << K << " x " << K << "x" << N << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "Efficiency: " << (tflops / 12.74 * 100.0) << "% of peak (12.74 TFLOPS)" << std::endl;
    
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
    std::cout << "SGEMM Performance Benchmark" << std::endl;
    std::cout << "Target: 9-10 TFLOPS (70-80% efficiency)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test various sizes
    benchmark_sgemm(512, 512, 512, 1000);
    benchmark_sgemm(1024, 1024, 1024, 500);
    benchmark_sgemm(2048, 2048, 2048, 100);
    benchmark_sgemm(4096, 4096, 4096, 50);    // Should use large kernel
    benchmark_sgemm(8192, 8192, 8192, 10);    // Should use large kernel
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
