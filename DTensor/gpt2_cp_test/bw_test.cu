// Raw device memory bandwidth test
#include <cstdio>
#include <cuda_runtime.h>

__global__ void copy_kernel(const float* __restrict__ src, float* __restrict__ dst, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void copy_kernel_vec4(const float4* __restrict__ src, float4* __restrict__ dst, int64_t n4) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n4; i += stride) {
        dst[i] = src[i];
    }
}

// Simulates adam-like access pattern: read 4 arrays, write 3
__global__ void adam_pattern_kernel(
    const float* __restrict__ g, float* __restrict__ p,
    float* __restrict__ m, float* __restrict__ v, int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float gi = g[i], pi = p[i], mi = m[i], vi = v[i];
        float m_new = 0.9f * mi + 0.1f * gi;
        float v_new = 0.95f * vi + 0.05f * gi * gi;
        m[i] = m_new;
        v[i] = v_new;
        p[i] = pi - 0.001f * (m_new / (sqrtf(v_new) + 1e-8f) + 0.1f * pi);
    }
}

int main() {
    cudaSetDevice(0);
    const int64_t N = 44350848;  // Same as GPT-2 total params
    
    float *d_a, *d_b, *d_c, *d_d;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMalloc(&d_d, N * sizeof(float));
    cudaMemset(d_a, 0, N * sizeof(float));
    cudaMemset(d_b, 0, N * sizeof(float));
    cudaMemset(d_c, 0, N * sizeof(float));
    cudaMemset(d_d, 0, N * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threads = 256;
    int blocks = 1024;
    
    printf("\n=== Raw GPU Memory Bandwidth Test (N=%lld = %.1fM) ===\n\n", (long long)N, N/1e6);
    
    // Warm up
    for (int i = 0; i < 5; i++) copy_kernel<<<blocks, threads>>>(d_a, d_b, N);
    cudaDeviceSynchronize();
    
    // Test 1: Simple copy (scalar)
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) copy_kernel<<<blocks, threads>>>(d_a, d_b, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop); ms /= 100;
    double bw = 2.0 * N * 4.0 / (ms * 1e-3) / 1e9;
    printf("  copy_kernel (scalar):  %.3f ms  =>  %.1f GB/s\n", ms, bw);
    
    // Test 2: Vectorized copy (float4)
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) copy_kernel_vec4<<<blocks, threads>>>((float4*)d_a, (float4*)d_b, N/4);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); ms /= 100;
    bw = 2.0 * N * 4.0 / (ms * 1e-3) / 1e9;
    printf("  copy_kernel (float4):  %.3f ms  =>  %.1f GB/s\n", ms, bw);
    
    // Test 3: Adam-like pattern (4 reads + 3 writes, NO binary search)
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) adam_pattern_kernel<<<blocks, threads>>>(d_a, d_b, d_c, d_d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); ms /= 100;
    bw = 7.0 * N * 4.0 / (ms * 1e-3) / 1e9;  // 4 reads + 3 writes
    printf("  adam_pattern (no bsearch): %.3f ms  =>  %.1f GB/s\n", ms, bw);
    
    // Test 4: Adam-like with more blocks
    blocks = 4096;
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) adam_pattern_kernel<<<blocks, threads>>>(d_a, d_b, d_c, d_d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); ms /= 100;
    bw = 7.0 * N * 4.0 / (ms * 1e-3) / 1e9;
    printf("  adam_pattern (4096 blks): %.3f ms  =>  %.1f GB/s\n", ms, bw);
    
    printf("\n  Expected RTX 3060 bandwidth: ~360 GB/s\n");
    printf("  If adam_pattern shows ~360 GB/s but multi_tensor_adam shows ~40 GB/s,\n");
    printf("  then the binary search overhead is the culprit.\n\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    return 0;
}
