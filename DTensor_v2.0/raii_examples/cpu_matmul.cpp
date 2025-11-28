#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>

// ============================================================================
// Matrix Multiplication Implementation
// ============================================================================

void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    // C = A @ B
    // A: [M, K], B: [K, N], C: [M, N]
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// WITHOUT RAII - Manual Memory Management
// ============================================================================

void cpu_matmul_without_raii() {
    std::cout << "\n=== WITHOUT RAII ===\n";
    
    int M = 512, K = 256, N = 1024;
    std::cout << "Computing matmul: [" << M << ", " << K << "] @ [" << K << ", " << N << "]\n";
    
    // ❌ Manual allocation for 3 arrays
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    
    // Initialize
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 2.0f;
    
    // Compute
    matmul_cpu(A, B, C, M, K, N);
    
    // Verify result
    float expected = K * 2.0f;  // Each element should be K * 2
    if (std::abs(C[0] - expected) > 1e-5) {
        // ❌ Must cleanup all 3 arrays before throwing!
        delete[] A;
        delete[] B;
        delete[] C;
        throw std::runtime_error("Wrong result");
    }
    
    std::cout << "Result verified: C[0] = " << C[0] << " (expected " << expected << ")\n";
    
    // ❌ Must cleanup all 3 arrays at end!
    delete[] A;
    delete[] B;
    delete[] C;
    
    std::cout << "Memory freed manually\n";
}

// ============================================================================
// WITH RAII - Automatic Memory Management
// ============================================================================

void cpu_matmul_with_raii() {
    std::cout << "\n=== WITH RAII ===\n";
    
    int M = 512, K = 256, N = 1024;
    std::cout << "Computing matmul: [" << M << ", " << K << "] @ [" << K << ", " << N << "]\n";
    
    // ✅ RAII: Auto allocation + cleanup for all 3 arrays
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N);
    
    // Compute
    matmul_cpu(A.data(), B.data(), C.data(), M, K, N);
    
    // Verify result
    float expected = K * 2.0f;
    if (std::abs(C[0] - expected) > 1e-5) {
        throw std::runtime_error("Wrong result");
        // ✅ All 3 arrays automatically cleaned up!
    }
    
    std::cout << "Result verified: C[0] = " << C[0] << " (expected " << expected << ")\n";
    
    // ✅ All 3 arrays automatically cleaned up when vectors go out of scope
    std::cout << "Memory will be freed automatically\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "RAII Example 2: CPU Matrix Multiplication\n";
    std::cout << "==========================================\n";
    
    try {
        cpu_matmul_without_raii();
    } catch (const std::exception& e) {
        std::cerr << "Caught exception (without RAII): " << e.what() << "\n";
    }
    
    try {
        cpu_matmul_with_raii();
    } catch (const std::exception& e) {
        std::cerr << "Caught exception (with RAII): " << e.what() << "\n";
    }
    
    std::cout << "\nProgram completed successfully.\n";
    std::cout << "\nNote: With 3 allocations, RAII eliminates 3 cleanup points in each error path.\n";
    return 0;
}
