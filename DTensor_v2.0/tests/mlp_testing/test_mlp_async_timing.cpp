
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
// Timing Result
// =============================================================================
struct AsyncTimingResult {
    int batch_size;
    int hidden_dim;
    
    // Blocking pattern: Compute → Sync → Next
    double blocking_total_us;
    
    // Async pattern: Compute overlaps with previous Sync
    double async_total_us;
    double async_launch_us;    // Time to launch async op
    double async_wait_us;      // Time to wait for completion
    
    // Speedup
    double speedup;
};

// =============================================================================
// Test 1: Blocking Pattern (baseline)
// =============================================================================
double benchmark_blocking_pattern(
    int BT, int C, int F_local,
    float* d_X, float* d_W1, float* d_H, float* d_W2, float* d_Y,
    cublasHandle_t handle,
    std::shared_ptr<ProcessGroupNCCL> pg,
    int warmup, int iters) {
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            F_local, BT, C, &alpha, d_W1, F_local, d_X, C, &beta, d_H, F_local));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            C, BT, F_local, &alpha, d_W2, C, d_H, F_local, &beta, d_Y, C));
        pg->all_reduce(d_Y, d_Y, BT * C, Dtype::Float32, sum, true);  // Blocking
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Measure
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            F_local, BT, C, &alpha, d_W1, F_local, d_X, C, &beta, d_H, F_local));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            C, BT, F_local, &alpha, d_W2, C, d_H, F_local, &beta, d_Y, C));
        pg->all_reduce(d_Y, d_Y, BT * C, Dtype::Float32, sum, true);  // Blocking
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    
    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, end));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    
    return (total_ms * 1000.0) / iters;  // Return avg per iteration in us
}

// =============================================================================
// Test 2: Async Pattern (overlapped)
// =============================================================================
struct AsyncResult {
    double total_us;
    double launch_us;
    double wait_us;
};

AsyncResult benchmark_async_pattern(
    int BT, int C, int F_local,
    float* d_X, float* d_W1, float* d_H, float* d_W2, float* d_Y,
    cublasHandle_t handle,
    std::shared_ptr<ProcessGroupNCCL> pg,
    int warmup, int iters) {
    
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
    cudaEvent_t start, after_launch, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&after_launch));
    CUDA_CHECK(cudaEventCreate(&end));
    
    double total_launch = 0, total_wait = 0;
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        // Compute
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            F_local, BT, C, &alpha, d_W1, F_local, d_X, C, &beta, d_H, F_local));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            C, BT, F_local, &alpha, d_W2, C, d_H, F_local, &beta, d_Y, C));
        
        // Launch async AllReduce
        cudaEvent_t launch_start, launch_end;
        CUDA_CHECK(cudaEventCreate(&launch_start));
        CUDA_CHECK(cudaEventCreate(&launch_end));
        
        CUDA_CHECK(cudaEventRecord(launch_start));
        auto work = pg->all_reduce_async(d_Y, d_Y, BT * C, Dtype::Float32, sum);
        CUDA_CHECK(cudaEventRecord(launch_end));
        CUDA_CHECK(cudaEventSynchronize(launch_end));
        
        float launch_ms;
        CUDA_CHECK(cudaEventElapsedTime(&launch_ms, launch_start, launch_end));
        total_launch += launch_ms * 1000.0;
        
        // Wait for completion
        cudaEvent_t wait_start, wait_end;
        CUDA_CHECK(cudaEventCreate(&wait_start));
        CUDA_CHECK(cudaEventCreate(&wait_end));
        
        CUDA_CHECK(cudaEventRecord(wait_start));
        work->wait();
        CUDA_CHECK(cudaEventRecord(wait_end));
        CUDA_CHECK(cudaEventSynchronize(wait_end));
        
        float wait_ms;
        CUDA_CHECK(cudaEventElapsedTime(&wait_ms, wait_start, wait_end));
        total_wait += wait_ms * 1000.0;
        
        CUDA_CHECK(cudaEventDestroy(launch_start));
        CUDA_CHECK(cudaEventDestroy(launch_end));
        CUDA_CHECK(cudaEventDestroy(wait_start));
        CUDA_CHECK(cudaEventDestroy(wait_end));
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    
    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, end));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(after_launch));
    CUDA_CHECK(cudaEventDestroy(end));
    
    AsyncResult result;
    result.total_us = (total_ms * 1000.0) / iters;
    result.launch_us = total_launch / iters;
    result.wait_us = total_wait / iters;
    
    return result;
}

// =============================================================================
// Main benchmark function
// =============================================================================
AsyncTimingResult benchmark_async_timing(
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
    
    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run benchmarks
    double blocking_us = benchmark_blocking_pattern(BT, C, F_local, d_X, d_W1, d_H, d_W2, d_Y, handle, pg, warmup, iters);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    AsyncResult async_res = benchmark_async_pattern(BT, C, F_local, d_X, d_W1, d_H, d_W2, d_Y, handle, pg, warmup, iters);
    
    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W1));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_Y));
    
    AsyncTimingResult result;
    result.batch_size = batch_size;
    result.hidden_dim = hidden_dim;
    result.blocking_total_us = blocking_us;
    result.async_total_us = async_res.total_us;
    result.async_launch_us = async_res.launch_us;
    result.async_wait_us = async_res.wait_us;
    result.speedup = blocking_us / async_res.total_us;
    
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
        std::cout << "\n=== Async Timing Test (TP=2, " << prop.name << ") ===\n";
        std::cout << std::setw(6) << "Batch" << std::setw(8) << "Hidden"
                  << std::setw(12) << "Blocking"
                  << std::setw(10) << "Async"
                  << std::setw(10) << "Launch"
                  << std::setw(10) << "Wait"
                  << std::setw(8) << "Speedup\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::vector<std::pair<int, int>> configs = {
        {4, 768},
        {8, 2048},
        {16, 4096},
    };
    
    for (auto& [bs, hdim] : configs) {
        auto result = benchmark_async_timing(bs, hdim, rank, mesh, pg, 3, 10);
        
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(0);
            std::cout << std::setw(6) << result.batch_size
                      << std::setw(8) << result.hidden_dim
                      << std::setw(12) << result.blocking_total_us
                      << std::setw(10) << result.async_total_us
                      << std::setw(10) << result.async_launch_us
                      << std::setw(10) << result.async_wait_us
                      << std::setprecision(2)
                      << std::setw(8) << result.speedup << "x\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        std::cout << "\n[PASS] Async timing test complete.\n";
    }
    
    MPI_Finalize();
    return 0;
}

// make test_mlp_async_timing
// mpirun -np 2 --allow-run-as-root ./test_mlp_async_timing
