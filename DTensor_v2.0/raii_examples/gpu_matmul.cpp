#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime.h>
#include "gpu_array.h"

// Forward declaration of kernel launcher
void launch_matmul_kernel(const float* d_A, const float* d_B, float* d_C,
                         int M, int K, int N);

// ============================================================================
// WITHOUT RAII - Manual GPU Memory Management
// ============================================================================

void gpu_matmul_without_raii() {
    std::cout << "\n=== WITHOUT RAII ===\n";
    
    int M = 512, K = 256, N = 1024;
    std::cout << "Computing GPU matmul: [" << M << ", " << K << "] @ [" << K << ", " << N << "]\n";
    
    // ❌ Manual CPU allocations
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    
    // ❌ Manual GPU allocations
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Initialize host data
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;
    
    // Copy to GPU
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    try {
        launch_matmul_kernel(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    } catch (const std::exception& e) {
        // ❌ Must cleanup ALL 6 allocations on error!
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw;
    }
    
    // Copy result back
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify
    float expected = K * 2.0f;
    if (std::abs(h_C[0] - expected) > 1e-3) {
        // ❌ Must cleanup ALL 6 allocations before throwing!
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Wrong result");
    }
    
    std::cout << "Result verified: C[0] = " << h_C[0] << " (expected " << expected << ")\n";
    
    // ❌ Must cleanup ALL 6 allocations at end!
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    std::cout << "All memory freed manually\n";
}

// ============================================================================
// WITH RAII - Automatic GPU Memory Management
// ============================================================================

void gpu_matmul_with_raii() {
    std::cout << "\n=== WITH RAII ===\n";
    
    int M = 512, K = 256, N = 1024;
    std::cout << "Computing GPU matmul: [" << M << ", " << K << "] @ [" << K << ", " << N << "]\n";
    
    // ✅ RAII: CPU memory
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N);
    
    // ✅ RAII: GPU memory (using our custom GPUArray wrapper)
    GPUArray<float> d_A(M * K);
    GPUArray<float> d_B(K * N);
    GPUArray<float> d_C(M * N);
    
    // Copy to GPU
    d_A.copyFrom(h_A.data());
    d_B.copyFrom(h_B.data());
    
    // Launch kernel
    launch_matmul_kernel(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    // ✅ If exception thrown, ALL 6 arrays automatically cleaned up!
    
    // Copy result back
    d_C.copyTo(h_C.data());
    
    // Verify
    float expected = K * 2.0f;
    if (std::abs(h_C[0] - expected) > 1e-3) {
        throw std::runtime_error("Wrong result");
        // ✅ ALL 6 arrays automatically cleaned up!
    }
    
    std::cout << "Result verified: C[0] = " << h_C[0] << " (expected " << expected << ")\n";
    
    // ✅ ALL 6 arrays automatically cleaned up when objects go out of scope
    std::cout << "All memory will be freed automatically\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "RAII Example 3: GPU Matrix Multiplication\n";
    std::cout << "==========================================\n";
    
    try {
        gpu_matmul_without_raii();
    } catch (const std::exception& e) {
        std::cerr << "Caught exception (without RAII): " << e.what() << "\n";
    }
    
    try {
        gpu_matmul_with_raii();
    } catch (const std::exception& e) {
        std::cerr << "Caught exception (with RAII): " << e.what() << "\n";
    }
    
    std::cout << "\nProgram completed successfully.\n";
    std::cout << "\nNote: With 6 allocations (3 CPU + 3 GPU), RAII becomes essential.\n";
    std::cout << "Without RAII: 6 cleanup calls in every error path.\n";
    std::cout << "With RAII: 0 manual cleanup calls - compiler handles it.\n";
    
    return 0;
}
