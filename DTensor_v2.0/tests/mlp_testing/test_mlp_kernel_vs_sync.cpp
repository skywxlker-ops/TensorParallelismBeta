
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
    double kernel_time_us;   // MatMul time
    double sync_time_us;     // AllReduce time
    double total_time_us;    // End-to-end
    double compute_ratio;    // kernel / total (%)
    double comm_ratio;       // sync / total (%)
};

// =============================================================================
// Benchmark using raw cuBLAS + NCCL (clean output)
// =============================================================================
TimingResult benchmark_kernel_vs_sync(
    int batch_size,
    int hidden_dim,
    int rank,
    std::shared_ptr<DeviceMesh> mesh,
    std::shared_ptr<ProcessGroupNCCL> pg,
    int warmup_iters,
    int measure_iters) {
    
    const int BT = batch_size * 512;  // batch * seq_len
    const int C = hidden_dim;
    const int F = hidden_dim * 4;     // FFN intermediate dim (sharded: F/2 per GPU)
    const int F_local = F / 2;        // Local shard size
    
    // Allocate GPU memory
    float *d_X, *d_W1, *d_H, *d_W2, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, BT * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1, C * F_local * sizeof(float)));  // Column-sharded
    CUDA_CHECK(cudaMalloc(&d_H, BT * F_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, F_local * C * sizeof(float)));  // Row-sharded
    CUDA_CHECK(cudaMalloc(&d_Y, BT * C * sizeof(float)));
    
    // Initialize with random data
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234 + rank);
    curandGenerateUniform(gen, d_X, BT * C);
    curandGenerateUniform(gen, d_W1, C * F_local);
    curandGenerateUniform(gen, d_W2, F_local * C);
    curandDestroyGenerator(gen);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // CUDA events for timing
    cudaEvent_t start, after_kernel, after_sync;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&after_kernel));
    CUDA_CHECK(cudaEventCreate(&after_sync));
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        // MatMul 1: X @ W1 = H  [BT, C] x [C, F_local] = [BT, F_local]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            F_local, BT, C, &alpha, d_W1, F_local, d_X, C, &beta, d_H, F_local));
        
        // MatMul 2: H @ W2 = Y  [BT, F_local] x [F_local, C] = [BT, C]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            C, BT, F_local, &alpha, d_W2, C, d_H, F_local, &beta, d_Y, C));
        
        // AllReduce
        pg->all_reduce(d_Y, d_Y, BT * C, Dtype::Float32, sum, true);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Measurement
    double total_kernel_time = 0.0;
    double total_sync_time = 0.0;
    
    for (int i = 0; i < measure_iters; i++) {
        // Start timing
        CUDA_CHECK(cudaEventRecord(start));
        
        // MatMul 1 + MatMul 2
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            F_local, BT, C, &alpha, d_W1, F_local, d_X, C, &beta, d_H, F_local));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            C, BT, F_local, &alpha, d_W2, C, d_H, F_local, &beta, d_Y, C));
        
        // Record after kernels
        CUDA_CHECK(cudaEventRecord(after_kernel));
        
        // AllReduce (blocking)
        pg->all_reduce(d_Y, d_Y, BT * C, Dtype::Float32, sum, true);
        
        // Record after sync
        CUDA_CHECK(cudaEventRecord(after_sync));
        CUDA_CHECK(cudaEventSynchronize(after_sync));
        
        // Calculate times
        float kernel_ms, sync_ms;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, after_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&sync_ms, after_kernel, after_sync));
        
        total_kernel_time += kernel_ms * 1000.0;  // to microseconds
        total_sync_time += sync_ms * 1000.0;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(after_kernel));
    CUDA_CHECK(cudaEventDestroy(after_sync));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W1));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_Y));
    
    // Calculate averages
    double avg_kernel = total_kernel_time / measure_iters;
    double avg_sync = total_sync_time / measure_iters;
    double avg_total = avg_kernel + avg_sync;
    
    TimingResult result;
    result.batch_size = batch_size;
    result.hidden_dim = hidden_dim;
    result.kernel_time_us = avg_kernel;
    result.sync_time_us = avg_sync;
    result.total_time_us = avg_total;
    result.compute_ratio = (avg_kernel / avg_total) * 100.0;
    result.comm_ratio = (avg_sync / avg_total) * 100.0;
    
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
        if (rank == 0) {
            std::cerr << "This test requires exactly 2 MPI processes (TP=2)." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Setup CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device_id = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaFree(0));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    // Setup process group
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    auto pg = init_process_group(world_size, rank);
    
    if (rank == 0) {
        std::cout << "\n=== Kernel vs Sync Time (TP=2, " << prop.name << ") ===\n";
        std::cout << std::setw(6) << "Batch" << std::setw(8) << "Hidden"
                  << std::setw(12) << "Kernel(us)" << std::setw(10) << "Sync(us)"
                  << std::setw(10) << "Compute%" << std::setw(8) << "Comm%\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test configurations
    std::vector<std::pair<int, int>> configs = {
        {4, 768},    // GPT-2 small
        {8, 2048},   // GPT-2 large
        {16, 4096},  // GPT-3 style
    };
    
    for (auto& [bs, hdim] : configs) {
        auto result = benchmark_kernel_vs_sync(bs, hdim, rank, mesh, pg, 3, 10);
        
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(0);
            std::cout << std::setw(6) << result.batch_size
                      << std::setw(8) << result.hidden_dim
                      << std::setw(12) << result.kernel_time_us
                      << std::setw(10) << result.sync_time_us
                      << std::setprecision(1)
                      << std::setw(10) << result.compute_ratio
                      << std::setw(8) << result.comm_ratio << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        std::cout << "\n[PASS] Kernel vs Sync test complete.\n";
    }
    
    MPI_Finalize();
    return 0;
}


// make test_mlp_kernel_vs_sync
// mpirun -np 2 --allow-run-as-root ./test_mlp_kernel_vs_sync 2>&1