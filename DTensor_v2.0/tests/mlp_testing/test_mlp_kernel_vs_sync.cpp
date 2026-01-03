
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>

#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include "memory/cachingAllocator.hpp"

using namespace OwnTensor;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// =============================================================================
// Timing Result Structure
// =============================================================================
struct TimingResult {
    int batch_size;
    int hidden_dim;
    
    // Sync pattern
    double kernel_time_us;
    double sync_time_us;
    double sync_total_us;
    double sync_compute_pct;
    double sync_comm_pct;
    
    // Async pattern
    double async_launch_us;
    double async_wait_us;
    double async_total_us;
    double async_compute_pct;
    double async_comm_pct;
};

// =============================================================================
// Benchmark Sync Pattern: Kernel → Blocking AllReduce → Next
// =============================================================================
void benchmark_sync_pattern(
    int BT, int C, int F_local,
    float* d_X, float* d_W1, float* d_H, float* d_W2, float* d_Y,
    cublasHandle_t handle,
    std::shared_ptr<ProcessGroupNCCL> pg,
    int warmup, int iters,
    double& out_kernel_us, double& out_sync_us) {
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            F_local, BT, C, &alpha, d_W1, F_local, d_X, C, &beta, d_H, F_local));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            C, BT, F_local, &alpha, d_W2, C, d_H, F_local, &beta, d_Y, C));
        pg->all_reduce(d_Y, d_Y, BT * C, Dtype::Float32, sum, true);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Measure
    cudaEvent_t start, after_kernel, after_sync;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&after_kernel));
    CUDA_CHECK(cudaEventCreate(&after_sync));
    
    double total_kernel = 0, total_sync = 0;
    
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            F_local, BT, C, &alpha, d_W1, F_local, d_X, C, &beta, d_H, F_local));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            C, BT, F_local, &alpha, d_W2, C, d_H, F_local, &beta, d_Y, C));
        
        CUDA_CHECK(cudaEventRecord(after_kernel));
        pg->all_reduce(d_Y, d_Y, BT * C, Dtype::Float32, sum, true);
        CUDA_CHECK(cudaEventRecord(after_sync));
        CUDA_CHECK(cudaEventSynchronize(after_sync));
        
        float kernel_ms, sync_ms;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, after_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&sync_ms, after_kernel, after_sync));
        
        total_kernel += kernel_ms * 1000.0;
        total_sync += sync_ms * 1000.0;
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(after_kernel));
    CUDA_CHECK(cudaEventDestroy(after_sync));
    
    out_kernel_us = total_kernel / iters;
    out_sync_us = total_sync / iters;
}

// =============================================================================
// Benchmark Async Pattern: Kernel → Async AllReduce → Wait
// =============================================================================
void benchmark_async_pattern(
    int BT, int C, int F_local,
    float* d_X, float* d_W1, float* d_H, float* d_W2, float* d_Y,
    cublasHandle_t handle,
    std::shared_ptr<ProcessGroupNCCL> pg,
    int warmup, int iters,
    double& out_kernel_us, double& out_launch_us, double& out_wait_us) {
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            F_local, BT, C, &alpha, d_W1, F_local, d_X, C, &beta, d_H, F_local));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            C, BT, F_local, &alpha, d_W2, C, d_H, F_local, &beta, d_Y, C));
        auto work = pg->all_reduce_async(d_Y, d_Y, BT * C, Dtype::Float32, sum);
        work->wait();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Measure
    cudaEvent_t start, after_kernel, after_launch, after_wait;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&after_kernel));
    CUDA_CHECK(cudaEventCreate(&after_launch));
    CUDA_CHECK(cudaEventCreate(&after_wait));
    
    double total_kernel = 0, total_launch = 0, total_wait = 0;
    
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            F_local, BT, C, &alpha, d_W1, F_local, d_X, C, &beta, d_H, F_local));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            C, BT, F_local, &alpha, d_W2, C, d_H, F_local, &beta, d_Y, C));
        
        CUDA_CHECK(cudaEventRecord(after_kernel));
        
        auto work = pg->all_reduce_async(d_Y, d_Y, BT * C, Dtype::Float32, sum);
        
        CUDA_CHECK(cudaEventRecord(after_launch));
        
        work->wait();
        
        CUDA_CHECK(cudaEventRecord(after_wait));
        CUDA_CHECK(cudaEventSynchronize(after_wait));
        
        float kernel_ms, launch_ms, wait_ms;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, after_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&launch_ms, after_kernel, after_launch));
        CUDA_CHECK(cudaEventElapsedTime(&wait_ms, after_launch, after_wait));
        
        total_kernel += kernel_ms * 1000.0;
        total_launch += launch_ms * 1000.0;
        total_wait += wait_ms * 1000.0;
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(after_kernel));
    CUDA_CHECK(cudaEventDestroy(after_launch));
    CUDA_CHECK(cudaEventDestroy(after_wait));
    
    out_kernel_us = total_kernel / iters;
    out_launch_us = total_launch / iters;
    out_wait_us = total_wait / iters;
}

// =============================================================================
// Main benchmark function
// =============================================================================
TimingResult benchmark_kernel_vs_sync(
    int batch_size,
    int hidden_dim,
    int rank,
    std::shared_ptr<DeviceMesh> mesh,
    std::shared_ptr<ProcessGroupNCCL> pg,
    int warmup, int iters) {
    
    const int BT = batch_size * 512;
    const int C = hidden_dim;
    const int F = hidden_dim * 4;
    const int F_local = F / 2;
    
    // Allocate
    float *d_X, *d_W1, *d_H, *d_W2, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, BT * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1, C * F_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, BT * F_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, F_local * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, BT * C * sizeof(float)));
    
    // Init random
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234 + rank);
    curandGenerateUniform(gen, d_X, BT * C);
    curandGenerateUniform(gen, d_W1, C * F_local);
    curandGenerateUniform(gen, d_W2, F_local * C);
    curandDestroyGenerator(gen);
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Sync pattern
    double sync_kernel, sync_time;
    benchmark_sync_pattern(BT, C, F_local, d_X, d_W1, d_H, d_W2, d_Y, handle, pg, warmup, iters, sync_kernel, sync_time);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Async pattern
    double async_kernel, async_launch, async_wait;
    benchmark_async_pattern(BT, C, F_local, d_X, d_W1, d_H, d_W2, d_Y, handle, pg, warmup, iters, async_kernel, async_launch, async_wait);
    
    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W1));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_Y));
    
    TimingResult result;
    result.batch_size = batch_size;
    result.hidden_dim = hidden_dim;
    
    // Sync results
    result.kernel_time_us = sync_kernel;
    result.sync_time_us = sync_time;
    result.sync_total_us = sync_kernel + sync_time;
    result.sync_compute_pct = (sync_kernel / result.sync_total_us) * 100.0;
    result.sync_comm_pct = (sync_time / result.sync_total_us) * 100.0;
    
    // Async results
    result.async_launch_us = async_launch;
    result.async_wait_us = async_wait;
    result.async_total_us = async_kernel + async_launch + async_wait;
    result.async_compute_pct = (async_kernel / result.async_total_us) * 100.0;
    result.async_comm_pct = ((async_launch + async_wait) / result.async_total_us) * 100.0;
    
    return result;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_size != 2) {
        if (rank == 0) std::cerr << "Requires TP=2\n";
        MPI_Finalize();
        return 1;
    }
    
    int device_id = rank % 2;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaFree(0));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    auto pg = init_process_group(world_size, rank);
    
    if (rank == 0) {
        std::cout << "\n=== Kernel vs Sync/Async Time (TP=2, " << prop.name << ") ===\n\n";
        
        // Sync header
        std::cout << "[SYNC Pattern] Kernel → Blocking AllReduce\n";
        std::cout << std::setw(6) << "Batch" << std::setw(8) << "Hidden"
                  << std::setw(10) << "Kernel" << std::setw(10) << "Sync"
                  << std::setw(10) << "Total" << std::setw(10) << "Comp%"
                  << std::setw(8) << "Comm%\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::vector<std::pair<int, int>> configs = {{4, 768}, {8, 2048}, {16, 4096}};
    std::vector<TimingResult> results;
    
    for (auto& [bs, hdim] : configs) {
        auto result = benchmark_kernel_vs_sync(bs, hdim, rank, mesh, pg, 3, 10);
        results.push_back(result);
        
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(0);
            std::cout << std::setw(6) << result.batch_size
                      << std::setw(8) << result.hidden_dim
                      << std::setw(10) << result.kernel_time_us
                      << std::setw(10) << result.sync_time_us
                      << std::setw(10) << result.sync_total_us
                      << std::setprecision(1)
                      << std::setw(10) << result.sync_compute_pct
                      << std::setw(8) << result.sync_comm_pct << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Async results
    if (rank == 0) {
        std::cout << "\n[ASYNC Pattern] Kernel → Async Launch → Wait\n";
        std::cout << std::setw(6) << "Batch" << std::setw(8) << "Hidden"
                  << std::setw(10) << "Launch" << std::setw(10) << "Wait"
                  << std::setw(10) << "Total" << std::setw(10) << "Comp%"
                  << std::setw(8) << "Comm%\n";
        
        for (auto& r : results) {
            std::cout << std::fixed << std::setprecision(0);
            std::cout << std::setw(6) << r.batch_size
                      << std::setw(8) << r.hidden_dim
                      << std::setw(10) << r.async_launch_us
                      << std::setw(10) << r.async_wait_us
                      << std::setw(10) << r.async_total_us
                      << std::setprecision(1)
                      << std::setw(10) << r.async_compute_pct
                      << std::setw(8) << r.async_comm_pct << "\n";
        }
        
        std::cout << "\n[PASS] Kernel vs Sync/Async test complete.\n";
    }
    
    MPI_Finalize();
    return 0;
}

// make test_mlp_kernel_vs_sync
// mpirun -np 2 --allow-run-as-root ./test_mlp_kernel_vs_sync
