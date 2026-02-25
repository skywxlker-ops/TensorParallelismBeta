#include "mycublas.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>

// Helper to check CUDA errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Simple CPU GEMM for verification
template<typename T>
void cpu_gemm_batched(
    int M, int N, int K,
    T alpha,
    const T* A, int lda, long long strideA,
    const T* B, int ldb, long long strideB,
    T beta,
    T* C, int ldc, long long strideC,
    int batchCount)
{
    for (int b = 0; b < batchCount; ++b) {
        const T* A_batch = A + b * strideA;
        const T* B_batch = B + b * strideB;
        T* C_batch = C + b * strideC;

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int p = 0; p < K; ++p) {
                    double a_val = static_cast<double>(A_batch[i * lda + p]);
                    double b_val = static_cast<double>(B_batch[p * ldb + j]);
                    sum += a_val * b_val;
                }
                double c_val = (beta == 0) ? 0.0 : static_cast<double>(C_batch[i * ldc + j]);
                C_batch[i * ldc + j] = static_cast<T>(static_cast<double>(alpha) * sum + static_cast<double>(beta) * c_val);
            }
        }
    }
}

// Specialization for BF16 (simulated as float for test simplicity if __nv_bfloat16 not available on host easily)
// Actually, MyBlas uses __nv_bfloat16 which is CUDA type. 
// For CPU test, we can use float arrays and cast during check.
// Or just replicate logic. 
// Since std::vector won't hold __nv_bfloat16 easily on host without headers, let's use float buffers and cast on device.

void test_sgemm_batched(mycublasHandle_t handle) {
    std::cout << "Testing SGEMM Strided Batched..." << std::endl;
    int M = 256, N = 256, K = 256;
    int batchCount = 4;
    float alpha = 1.0f, beta = 0.0f;

    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;

    std::vector<float> h_A(batchCount * strideA);
    std::vector<float> h_B(batchCount * strideB);
    std::vector<float> h_C(batchCount * strideC);
    std::vector<float> h_C_ref(batchCount * strideC);

    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < h_C.size(); ++i) h_C[i] = 0.0f;
    h_C_ref = h_C;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(float), cudaMemcpyHostToDevice));

    mycublasSgemmStridedBatched(handle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C, N, strideC, batchCount);
    CHECK_CUDA(cudaDeviceSynchronize());

    cpu_gemm_batched(M, N, K, alpha, h_A.data(), K, strideA, h_B.data(), N, strideB, beta, h_C_ref.data(), N, strideC, batchCount);

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (size_t i = 0; i < h_C.size(); ++i) {
        float diff = std::abs(h_C[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Max diff: " << max_diff << (max_diff < 1e-1 ? " [PASS]" : " [FAIL]") << std::endl;
    if (max_diff > 1e-1) {
        int count = 0;
        for (size_t i = 0; i < h_C.size(); ++i) {
             float diff = std::abs(h_C[i] - h_C_ref[i]);
             if (diff > 1e-3) {
                 std::cout << "Mismatch at index " << i << ": GPU=" << h_C[i] << ", CPU=" << h_C_ref[i] << ", Diff=" << diff << std::endl;
                 count++;
                 if (count > 5) break; 
             }
        }
    }
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// Helper for Half
// We can use array of __half on host if cuda_fp16.h is available.
void test_hgemm_batched(mycublasHandle_t handle) {
    std::cout << "Testing HGEMM Strided Batched..." << std::endl;
    int M = 256, N = 256, K = 256;
    int batchCount = 4;
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;

    std::vector<__half> h_A(batchCount * strideA);
    std::vector<__half> h_B(batchCount * strideB);
    std::vector<__half> h_C(batchCount * strideC);
    std::vector<__half> h_C_ref(batchCount * strideC);

    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    for (size_t i = 0; i < h_C.size(); ++i) h_C[i] = __float2half(0.0f);
    h_C_ref = h_C;

    __half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(__half), cudaMemcpyHostToDevice));

    mycublasHgemmStridedBatched(handle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C, N, strideC, batchCount);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Compute Ref on CPU (convert to float for math)
    // Simplified: reuse cpu_gemm_batched with float arrays
    std::vector<float> f_A(h_A.size()), f_B(h_B.size()), f_C(h_C_ref.size());
    for(size_t i=0; i<h_A.size(); ++i) f_A[i] = __half2float(h_A[i]);
    for(size_t i=0; i<h_B.size(); ++i) f_B[i] = __half2float(h_B[i]);
    cpu_gemm_batched(M, N, K, 1.0f, f_A.data(), K, strideA, f_B.data(), N, strideB, 0.0f, f_C.data(), N, strideC, batchCount);

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (size_t i = 0; i < h_C.size(); ++i) {
        float val = __half2float(h_C[i]);
        float ref = f_C[i];
        float diff = std::abs(val - ref);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Max diff: " << max_diff << (max_diff < 1e-1 ? " [PASS]" : " [FAIL]") << std::endl;
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// BF16 Test
void test_bgemm_batched(mycublasHandle_t handle) {
    std::cout << "Testing BGEMM Strided Batched..." << std::endl;
    int M = 256, N = 256, K = 256;
    int batchCount = 4;
    __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    __nv_bfloat16 beta = __float2bfloat16(0.0f);

    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;

    // Use float on host for simplicity validation
    std::vector<float> h_A(batchCount * strideA);
    std::vector<float> h_B(batchCount * strideB);
    std::vector<float> h_C(batchCount * strideC);
    std::vector<float> h_C_ref(batchCount * strideC);

    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    
    // Allocate device memory 
    // We need to convert float array to __nv_bfloat16 array on device.
    // Since we don't have a kernel to convert, let's copy float to device, then run a conversion kernel?
    // Or just implement a host helper to convert if we can include <cuda_bf16.h> on host.
    // Modern CUDA allows __nv_bfloat16 on host.
    
    std::vector<__nv_bfloat16> bf_A(h_A.size());
    std::vector<__nv_bfloat16> bf_B(h_B.size());
    std::vector<__nv_bfloat16> bf_C(h_C.size());
    
    for(size_t i=0; i<h_A.size(); ++i) bf_A[i] = __float2bfloat16(h_A[i]);
    for(size_t i=0; i<h_B.size(); ++i) bf_B[i] = __float2bfloat16(h_B[i]);
    for(size_t i=0; i<h_C.size(); ++i) bf_C[i] = __float2bfloat16(0.0f);

    __nv_bfloat16 *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bf_A.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, bf_B.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, bf_C.size() * sizeof(__nv_bfloat16)));

    CHECK_CUDA(cudaMemcpy(d_A, bf_A.data(), bf_A.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, bf_B.data(), bf_B.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, bf_C.data(), bf_C.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    mycublasBgemmStridedBatched(handle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C, N, strideC, batchCount);
    CHECK_CUDA(cudaDeviceSynchronize());

    cpu_gemm_batched(M, N, K, 1.0f, h_A.data(), K, strideA, h_B.data(), N, strideB, 0.0f, h_C_ref.data(), N, strideC, batchCount);

    CHECK_CUDA(cudaMemcpy(bf_C.data(), d_C, bf_C.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < bf_C.size(); ++i) {
        float val = __bfloat162float(bf_C[i]);
        float ref = h_C_ref[i];
        float diff = std::abs(val - ref);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Max diff: " << max_diff << (max_diff < 0.5f ? " [PASS]" : " [FAIL]") << std::endl; // BF16 ranges slightly larger error
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// FP64 Test
void test_dgemm_batched(mycublasHandle_t handle) {
    std::cout << "Testing DGEMM Strided Batched..." << std::endl;
    int M = 128, N = 128, K = 128; // Smaller for FP64 speed
    int batchCount = 4;
    double alpha = 1.0;
    double beta = 0.0;

    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;

    std::vector<double> h_A(batchCount * strideA);
    std::vector<double> h_B(batchCount * strideB);
    std::vector<double> h_C(batchCount * strideC);
    std::vector<double> h_C_ref(batchCount * strideC);

    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = static_cast<double>(rand()) / RAND_MAX;

    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(double), cudaMemcpyHostToDevice));

    mycublasDgemmStridedBatched(handle, M, N, K, alpha, d_A, K, strideA, d_B, N, strideB, beta, d_C, N, strideC, batchCount);
    CHECK_CUDA(cudaDeviceSynchronize());

    cpu_gemm_batched(M, N, K, alpha, h_A.data(), K, strideA, h_B.data(), N, strideB, beta, h_C_ref.data(), N, strideC, batchCount);

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(double), cudaMemcpyDeviceToHost));

    double max_diff = 0.0;
    for (size_t i = 0; i < h_C.size(); ++i) {
        double diff = std::abs(h_C[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Max diff: " << max_diff << (max_diff < 1e-10 ? " [PASS]" : " [FAIL]") << std::endl;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    mycublasHandle_t handle;
    mycublasCreate(&handle);

    // Relaxed tolerances for TF32 and FP16 due to accumulation differences
    std::cout << "NOTE: Tolerances relaxed for TF32/FP16 due to accumulation differences." << std::endl;
    
    test_sgemm_batched(handle);
    test_hgemm_batched(handle);
    test_bgemm_batched(handle);
    test_dgemm_batched(handle);

    mycublasDestroy(handle);
    return 0;
}
