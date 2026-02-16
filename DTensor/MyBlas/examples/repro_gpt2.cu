#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "mycublas.h"

void check_matmul(mycublasHandle_t handle, int M, int N, int K, int batchCount, const std::string& name) {
    std::cout << "Checking " << name << " (M=" << M << ", N=" << N << ", K=" << K << ", batches=" << batchCount << ")..." << std::endl;
    
    long long int strideA = (long long int)M * K;
    long long int strideB = 0; // standard for weight
    long long int strideC = (long long int)M * N;

    std::vector<float> h_A(M * K * batchCount, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_C(M * N * batchCount, 0.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, h_A.size() * sizeof(float));
    cudaMalloc(&d_B, h_B.size() * sizeof(float));
    cudaMalloc(&d_C, h_C.size() * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, h_C.size() * sizeof(float));

    mycublasSgemmStridedBatched(handle, M, N, K, 1.0f, d_A, K, strideA, d_B, N, strideB, 0.0f, d_C, N, strideC, batchCount);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "!!! " << name << " FAILED with CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check first few elements (assuming all input 1s, output should be K)
    bool correct = true;
    for (int b = 0; b < batchCount; ++b) {
        for (int i = 0; i < 100 && i < M*N; ++i) {
            if (std::abs(h_C[b*M*N + i] - (float)K) > 1e-3) {
                std::cout << "  Mismatch at batch " << b << ", idx " << i << ": " << h_C[b*M*N + i] << " != " << K << std::endl;
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    if (correct) {
        std::cout << "--- " << name << " PASSED (Values correct) ---" << std::endl;
    } else {
        std::cout << "!!! " << name << " FAILED value check" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    mycublasHandle_t handle;
    mycublasCreate(&handle);

    // Attention Matmul (Simplified to 3D)
    // [48, 1024, 64] * [48, 64, 1024] -> Using strideB = K*N
    {
        int M=1024, N=1024, K=64, batchCount=48;
        std::cout << "Checking Attention..." << std::endl;
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, (size_t)M*K*batchCount*4);
        cudaMalloc(&d_B, (size_t)K*N*batchCount*4);
        cudaMalloc(&d_C, (size_t)M*N*batchCount*4);
        mycublasSgemmStridedBatched(handle, M, N, K, 1.0f, d_A, K, (long long)M*K, d_B, N, (long long)K*N, 0.0f, d_C, N, (long long)M*N, batchCount);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) std::cout << "Attention FAILED: " << cudaGetErrorString(err) << std::endl;
        else std::cout << "Attention PASSED" << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // MLP Matmul
    check_matmul(handle, 1024, 3072, 768, 4, "MLP Down");

    // Final Projection (Large N)
    check_matmul(handle, 4096, 50304, 768, 1, "Final Projection");

    mycublasDestroy(handle);
    return 0;
}
