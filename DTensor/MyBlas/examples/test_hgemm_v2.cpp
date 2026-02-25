#include "mycublas.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <functional>

// Helper to check CUDA errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

double benchmark_kernel(
    std::function<void()> kernel_func,
    int warmup = 3,
    int iterations = 10)
{
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel_func();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        kernel_func();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count() / iterations;
}

void test_correctness() {
    std::cout << "\n=== CORRECTNESS TEST ===" << std::endl;
    
    int M = 256, N = 256, K = 256;
    int batchCount = 4;
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;
    
    // Allocate host memory
    std::vector<__half> h_A(batchCount * strideA);
    std::vector<__half> h_B(batchCount * strideB);
    std::vector<__half> h_C_v2(batchCount * strideC);
    std::vector<__half> h_C_cublas(batchCount * strideC);
    
    // Initialize with random values
    for (size_t i = 0; i < h_A.size(); ++i) {
        h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    for (size_t i = 0; i < h_B.size(); ++i) {
        h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    
    // Allocate device memory
    __half *d_A, *d_B, *d_C_v2, *d_C_cublas;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C_v2, h_C_v2.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C_cublas, h_C_cublas.size() * sizeof(__half)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C_v2, 0, h_C_v2.size() * sizeof(__half)));
    CHECK_CUDA(cudaMemset(d_C_cublas, 0, h_C_cublas.size() * sizeof(__half)));
    
    // Run V2 kernel
    mycublasHandle_t myblas_handle;
    mycublasCreate(&myblas_handle);
    mycublasHgemmStridedBatchedV2(myblas_handle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C_v2, N, strideC, batchCount);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Run cuBLAS reference
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasHgemmStridedBatched(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N, strideB,
        d_A, K, strideA,
        &beta,
        d_C_cublas, N, strideC,
        batchCount
    ));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_C_v2.data(), d_C_v2, h_C_v2.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C_cublas, h_C_cublas.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    
    // Compare results
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int mismatch_count = 0;
    
    for (size_t i = 0; i < h_C_v2.size(); ++i) {
        float v2_val = __half2float(h_C_v2[i]);
        float ref_val = __half2float(h_C_cublas[i]);
        float diff = std::abs(v2_val - ref_val);
        
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
        
        if (diff > 0.1f) {
            if (mismatch_count < 5) {
                std::cout << "  Mismatch at index " << i 
                         << ": V2=" << v2_val 
                         << ", cuBLAS=" << ref_val 
                         << ", Diff=" << diff << std::endl;
            }
            mismatch_count++;
        }
    }
    
    avg_diff /= h_C_v2.size();
    
    std::cout << "Max difference: " << max_diff << std::endl;
    std::cout << "Avg difference: " << avg_diff << std::endl;
    std::cout << "Mismatches (>0.1): " << mismatch_count << "/" << h_C_v2.size() << std::endl;
    
    if (max_diff < 0.1f) {
        std::cout << "✓ CORRECTNESS TEST PASSED" << std::endl;
    } else {
        std::cout << "✗ CORRECTNESS TEST FAILED" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_v2);
    cudaFree(d_C_cublas);
    mycublasDestroy(myblas_handle);
    cublasDestroy(cublas_handle);
}

void benchmark_performance() {
    std::cout << "\n=== PERFORMANCE BENCHMARK ===" << std::endl;
    
    struct TestConfig {
        int M, N, K, batchCount;
        std::string name;
    };
    
    std::vector<TestConfig> configs = {
        {1024, 1024, 1024, 1, "1K x 1K x 1K, batch=1"},
        {2048, 2048, 2048, 1, "2K x 2K x 2K, batch=1"},
        {4096, 4096, 1024, 1, "4K x 4K x 1K, batch=1"},
        {4096, 4096, 4096, 1, "4K x 4K x 4K, batch=1"},
        {2048, 2048, 2048, 8, "2K x 2K x 2K, batch=8"},
        {1024, 1024, 1024, 16, "1K x 1K x 1K, batch=16"},
    };
    
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    mycublasHandle_t myblas_handle;
    mycublasCreate(&myblas_handle);
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n" << std::setw(30) << "Config" 
              << std::setw(15) << "V2 (ms)"
              << std::setw(15) << "cuBLAS (ms)"
              << std::setw(15) << "V2 TFLOPS"
              << std::setw(15) << "cuBLAS TFLOPS"
              << std::setw(15) << "Speedup"
              << std::endl;
    std::cout << std::string(105, '-') << std::endl;
    
    for (const auto& config : configs) {
        int M = config.M;
        int N = config.N;
        int K = config.K;
        int batchCount = config.batchCount;
        
        long long strideA = M * K;
        long long strideB = K * N;
        long long strideC = M * N;
        
        // Allocate device memory
        __half *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, batchCount * strideA * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_B, batchCount * strideB * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_C, batchCount * strideC * sizeof(__half)));
        
        // Benchmark V2
        auto v2_time = benchmark_kernel([&]() {
            mycublasHgemmStridedBatchedV2(myblas_handle, M, N, K, alpha, 
                                         d_A, K, strideA, 
                                         d_B, N, strideB, 
                                         beta, d_C, N, strideC, batchCount);
        });
        
        // Benchmark cuBLAS
        auto cublas_time = benchmark_kernel([&]() {
            cublasHgemmStridedBatched(cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha,
                                     d_B, N, strideB,
                                     d_A, K, strideA,
                                     &beta,
                                     d_C, N, strideC,
                                     batchCount);
        });
        
        // Calculate TFLOPS
        double flops = 2.0 * M * N * K * batchCount;
        double v2_tflops = flops / (v2_time * 1e12);
        double cublas_tflops = flops / (cublas_time * 1e12);
        double speedup = cublas_time / v2_time;
        
        std::cout << std::setw(30) << config.name
                  << std::setw(15) << (v2_time * 1000)
                  << std::setw(15) << (cublas_time * 1000)
                  << std::setw(15) << v2_tflops
                  << std::setw(15) << cublas_tflops
                  << std::setw(15) << speedup << "x"
                  << std::endl;
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    mycublasDestroy(myblas_handle);
    cublasDestroy(cublas_handle);
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "HGEMM Strided Batched V2 Test & Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Get device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "\nDevice: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "CUDA Cores (SMs x cores/SM): " << prop.multiProcessorCount << " x " << std::endl;
    std::cout << "Peak FP16 (estimated): ~50 TFLOPS" << std::endl;
    
    // Run tests
    test_correctness();
    benchmark_performance();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "All tests completed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
