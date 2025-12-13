#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>

// Case 1: Current DTensor (wrapper)
#include "tensor/dtensor.h"

// Case 2: Native DTensor (in OwnTensor namespace)
#include "tensor/dtensor_native.h"

#include "process_group/process_group.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

struct BenchmarkResult {
    double init_time_ms;
    double matmul_time_ms;
    size_t memory_mb;
};

size_t get_gpu_memory_used() {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return (total - free) / (1024 * 1024);  // MB
}

// Benchmark Case 1: Current wrapper architecture
BenchmarkResult benchmark_wrapper(int rank, std::shared_ptr<DeviceMesh> mesh,
                                    std::shared_ptr<ProcessGroup> pg,
                                    int M, int K, int N) {
    const int outer_iterations = 100;
    double total_init_time = 0.0;
    double total_matmul_time = 0.0;
    size_t final_memory = 0;
    
    for (int iter = 0; iter < outer_iterations; ++iter) {
        // Measure initialization time
        auto init_start = Clock::now();
        
        DTensor X(mesh, pg);
        DTensor W(mesh, pg);
        
        // Create replicated X [M, K] and column-sharded W [K, N/2]
        std::vector<int> X_shape = {M, K};
        std::vector<int> W_shape = {K, N};
        
        Layout X_layout = Layout::replicated(mesh, X_shape);
        Layout W_layout(mesh, W_shape, ShardingType::SHARDED, 1);
        
        // Prepare data
        std::vector<float> X_data(M * K, 1.0f);
        std::vector<float> W_data(K * (N / 2), 1.0f);
        
        X.setData(X_data, X_layout);
        W.setData(W_data, W_layout);
        
        auto init_end = Clock::now();
        total_init_time += Duration(init_end - init_start).count();
        
        // Warmup
        if (iter == 0) {
            DTensor Y = X.matmul(W);
            auto temp = Y.getData();
        }
        
        // Measure matmul time (multiple iterations)
        const int inner_iterations = 10;
        auto matmul_start = Clock::now();
        for (int i = 0; i < inner_iterations; ++i) {
            DTensor Y_temp = X.matmul(W);
        }
        cudaDeviceSynchronize();
        auto matmul_end = Clock::now();
        total_matmul_time += Duration(matmul_end - matmul_start).count() / inner_iterations;
        
        // Measure memory on last iteration
        if (iter == outer_iterations - 1) {
            final_memory = get_gpu_memory_used();
        }
    }
    
    BenchmarkResult result;
    result.init_time_ms = total_init_time / outer_iterations;
    result.matmul_time_ms = total_matmul_time / outer_iterations;
    result.memory_mb = final_memory;
    
    return result;
}

// Benchmark Case 2: Native OwnTensor architecture
BenchmarkResult benchmark_native(int rank, std::shared_ptr<DeviceMesh> mesh,
                                   std::shared_ptr<ProcessGroup> pg,
                                   int M, int K, int N) {
    const int outer_iterations = 100;
    double total_init_time = 0.0;
    double total_matmul_time = 0.0;
    size_t final_memory = 0;
    
    for (int iter = 0; iter < outer_iterations; ++iter) {
        // Measure initialization time
        auto init_start = Clock::now();
        
        OwnTensor::DTensorNative X(mesh, pg);
        OwnTensor::DTensorNative W(mesh, pg);
        
        // Create replicated X [M, K] and column-sharded W [K, N/2]
        std::vector<int> X_shape = {M, K};
        std::vector<int> W_shape = {K, N};
        
        Layout X_layout = Layout::replicated(mesh, X_shape);
        Layout W_layout(mesh, W_shape, ShardingType::SHARDED, 1);
        
        // Prepare data
        std::vector<float> X_data(M * K, 1.0f);
        std::vector<float> W_data(K * (N / 2), 1.0f);
        
        X.setData(X_data, X_layout);
        W.setData(W_data, W_layout);
        
        auto init_end = Clock::now();
        total_init_time += Duration(init_end - init_start).count();
        
        // Warmup
        if (iter == 0) {
            OwnTensor::DTensorNative Y = X.matmul(W);
            auto temp = Y.getData();
        }
        
        // Measure matmul time (multiple iterations)
        const int inner_iterations = 10;
        auto matmul_start = Clock::now();
        for (int i = 0; i < inner_iterations; ++i) {
            OwnTensor::DTensorNative Y_temp = X.matmul(W);
        }
        cudaDeviceSynchronize();
        auto matmul_end = Clock::now();
        total_matmul_time += Duration(matmul_end - matmul_start).count() / inner_iterations;
        
        // Measure memory on last iteration
        if (iter == outer_iterations - 1) {
            final_memory = get_gpu_memory_used();
        }
    }
    
    BenchmarkResult result;
    result.init_time_ms = total_init_time / outer_iterations;
    result.matmul_time_ms = total_matmul_time / outer_iterations;
    result.memory_mb = final_memory;
    
    return result;
}

void run_benchmark(int rank, std::shared_ptr<DeviceMesh> mesh,
                    std::shared_ptr<ProcessGroup> pg,
                    int M, int K, int N) {
    if (rank == 0) {
        std::cout << "\n[Matrix size: " << M << "x" << K << " @ " << K << "x" << N << "]" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Benchmark wrapper version
    auto wrapper_result = benchmark_wrapper(rank, mesh, pg, M, K, N);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Benchmark native version (will reuse some allocations, but measure absolute peak)
    auto native_result = benchmark_native(rank, mesh, pg, M, K, N);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(3);
        
        // Calculate totals
        double wrapper_total = wrapper_result.init_time_ms + wrapper_result.matmul_time_ms;
        double native_total = native_result.init_time_ms + native_result.matmul_time_ms;
        
        std::cout << "  Case 1 (Wrapper):  init=" << wrapper_result.init_time_ms 
                  << "ms, matmul=" << wrapper_result.matmul_time_ms 
                  << "ms, total=" << wrapper_total
                  << "ms, mem=" << wrapper_result.memory_mb << "MB" << std::endl;
        std::cout << "  Case 2 (Native):   init=" << native_result.init_time_ms 
                  << "ms, matmul=" << native_result.matmul_time_ms 
                  << "ms, total=" << native_total
                  << "ms, mem=" << native_result.memory_mb << "MB" << std::endl;
        
        // Calculate speedup
        double init_speedup = wrapper_result.init_time_ms / native_result.init_time_ms;
        double matmul_speedup = wrapper_result.matmul_time_ms / native_result.matmul_time_ms;
        double total_speedup = wrapper_total / native_total;
        
        std::cout << "  Speedup:           init=" << init_speedup << "x"
                  << ", matmul=" << matmul_speedup << "x"
                  << ", total=" << total_speedup << "x" << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "This benchmark requires exactly 2 MPI processes." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Setup NCCL
    ncclUniqueId nccl_id;
    if (rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }
    MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    std::vector<int> mesh_shape = {world_size};
    auto mesh = std::make_shared<DeviceMesh>(mesh_shape);
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);

    CUDA_CHECK(cudaSetDevice(rank));

    if (rank == 0) {
        std::cout << "\nArchitecture Benchmark: Wrapper vs Native\n";
        std::cout << "==========================================\n";
        std::cout << "(Averaged over 100 iterations)\n";
    }

    // Run benchmarks with different matrix sizes
    run_benchmark(rank, mesh, pg, 512, 512, 512);      // Small
    run_benchmark(rank, mesh, pg, 1024, 1024, 1024);   // Medium
    run_benchmark(rank, mesh, pg, 2048, 2048, 2048);   // Large
    run_benchmark(rank, mesh, pg, 4096, 4096, 4096);   // Very Large
    run_benchmark(rank, mesh, pg, 8192, 8192, 8192);   // Huge

    if (rank == 0) {
        std::cout << "\nDone.\n";
    }

    MPI_Finalize();
    return 0;
}