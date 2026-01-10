#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <string>

#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
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

struct LayoutBenchmarkResult {
    std::string transformation_name;
    double avg_time_ms;
    double data_moved_gb;
    double bandwidth_gbps;
    size_t memory_before_mb;
    size_t memory_after_mb;
    size_t memory_peak_mb;
};

size_t get_gpu_memory_used() {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return (total - free) / (1024 * 1024);  // MB
}

// Benchmark: Replicated → Row-sharded
LayoutBenchmarkResult benchmark_replicate_to_row_shard(
    int rank, std::shared_ptr<DeviceMesh> mesh,
    std::shared_ptr<ProcessGroupNCCL> pg,
    int M, int N) {
    
    const int warmup_iters = 100;
    const int measure_iters = 500;
    double total_time = 0.0;
    size_t mem_before = 0, mem_after = 0, mem_peak = 0;
    
    for (int iter = 0; iter < warmup_iters + measure_iters; ++iter) {
        size_t iter_mem_before = get_gpu_memory_used();
        
        // Create replicated tensor
        DTensor X(mesh, pg);
        std::vector<int64_t> shape = {M, N};
        Layout replicated_layout = Layout::replicated(*mesh, shape);
        
        std::vector<float> data(M * N, 1.0f);
        X.setData(data, replicated_layout);
        
        size_t iter_mem_peak = get_gpu_memory_used();
        
        // Measure transformation time
        auto start = Clock::now();
        X.shard(0);  // Shard along dimension 0 (rows)
        cudaDeviceSynchronize();
        auto end = Clock::now();
        
        size_t iter_mem_after = get_gpu_memory_used();
        
        // Only record measurement iterations
        if (iter >= warmup_iters) {
            total_time += Duration(end - start).count();
            mem_before = iter_mem_before;
            mem_after = iter_mem_after;
            mem_peak = std::max(mem_peak, iter_mem_peak);
        }
    }
    
    double avg_time = total_time / measure_iters;
    
    // Calculate data movement (broadcast full tensor, then redistribute)
    // In practice, shard() does: broadcast from root + local extraction
    double data_moved_gb = (M * N * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gbps = data_moved_gb / (avg_time / 1000.0);  // GB/s
    
    LayoutBenchmarkResult result;
    result.transformation_name = "Replicated → Row-sharded";
    result.avg_time_ms = avg_time;
    result.data_moved_gb = data_moved_gb;
    result.bandwidth_gbps = bandwidth_gbps;
    result.memory_before_mb = mem_before;
    result.memory_after_mb = mem_after;
    result.memory_peak_mb = mem_peak;
    
    return result;
}

// Benchmark: Replicated → Column-sharded
LayoutBenchmarkResult benchmark_replicate_to_col_shard(
    int rank, std::shared_ptr<DeviceMesh> mesh,
    std::shared_ptr<ProcessGroupNCCL> pg,
    int M, int N) {
    
    const int warmup_iters = 100;
    const int measure_iters = 500;
    double total_time = 0.0;
    size_t mem_before = 0, mem_after = 0, mem_peak = 0;
    
    for (int iter = 0; iter < warmup_iters + measure_iters; ++iter) {
        size_t iter_mem_before = get_gpu_memory_used();
        
        // Create replicated tensor
        DTensor X(mesh, pg);
        std::vector<int64_t> shape = {M, N};
        Layout replicated_layout = Layout::replicated(*mesh, shape);
        
        std::vector<float> data(M * N, 1.0f);
        X.setData(data, replicated_layout);
        
        size_t iter_mem_peak = get_gpu_memory_used();
        
        // Measure transformation time
        auto start = Clock::now();
        X.shard(1);  // Shard along dimension 1 (columns)
        cudaDeviceSynchronize();
        auto end = Clock::now();
        
        size_t iter_mem_after = get_gpu_memory_used();
        
        // Only record measurement iterations
        if (iter >= warmup_iters) {
            total_time += Duration(end - start).count();
            mem_before = iter_mem_before;
            mem_after = iter_mem_after;
            mem_peak = std::max(mem_peak, iter_mem_peak);
        }
    }
    
    double avg_time = total_time / measure_iters;
    double data_moved_gb = (M * N * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gbps = data_moved_gb / (avg_time / 1000.0);
    
    LayoutBenchmarkResult result;
    result.transformation_name = "Replicated → Col-sharded";
    result.avg_time_ms = avg_time;
    result.data_moved_gb = data_moved_gb;
    result.bandwidth_gbps = bandwidth_gbps;
    result.memory_before_mb = mem_before;
    result.memory_after_mb = mem_after;
    result.memory_peak_mb = mem_peak;
    
    return result;
}

// Benchmark: Row-sharded → Column-sharded (redistribute)
LayoutBenchmarkResult benchmark_row_to_col_shard(
    int rank, std::shared_ptr<DeviceMesh> mesh,
    std::shared_ptr<ProcessGroupNCCL> pg,
    int M, int N) {
    
    const int warmup_iters = 100;
    const int measure_iters = 500;
    double total_time = 0.0;
    size_t mem_before = 0, mem_after = 0, mem_peak = 0;
    
    for (int iter = 0; iter < warmup_iters + measure_iters; ++iter) {
        // Start with row-sharded tensor
        DTensor X(mesh, pg);
        std::vector<int64_t> shape = {M, N};
        Layout row_sharded_layout(*mesh, shape, 0);
        
        // Each rank has M/world_size rows
        int world_size = mesh->size();
        std::vector<float> local_data((M / world_size) * N, 1.0f);
        X.setData(local_data, row_sharded_layout);
        
        size_t iter_mem_before = get_gpu_memory_used();
        
        // Measure redistribution: Row-shard → Replicated → Col-shard
        auto start = Clock::now();
        
        // Step 1: AllGather to replicate
        X.replicate(0);
        
        size_t iter_mem_peak = get_gpu_memory_used();
        
        // Step 2: Shard along columns
        X.shard(1);
        cudaDeviceSynchronize();
        auto end = Clock::now();
        
        size_t iter_mem_after = get_gpu_memory_used();
        
        // Only record measurement iterations
        if (iter >= warmup_iters) {
            total_time += Duration(end - start).count();
            mem_before = iter_mem_before;
            mem_after = iter_mem_after;
            mem_peak = std::max(mem_peak, iter_mem_peak);
        }
    }
    
    double avg_time = total_time / measure_iters;
    
    // Data movement: AllGather (full tensor assembled) + local extraction
    double data_moved_gb = (M * N * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gbps = data_moved_gb / (avg_time / 1000.0);
    
    LayoutBenchmarkResult result;
    result.transformation_name = "Row-shard → Col-shard";
    result.avg_time_ms = avg_time;
    result.data_moved_gb = data_moved_gb;
    result.bandwidth_gbps = bandwidth_gbps;
    result.memory_before_mb = mem_before;
    result.memory_after_mb = mem_after;
    result.memory_peak_mb = mem_peak;
    
    return result;
}

// Benchmark: Sharded → Replicated (via allGather)
LayoutBenchmarkResult benchmark_shard_to_replicated(
    int rank, std::shared_ptr<DeviceMesh> mesh,
    std::shared_ptr<ProcessGroupNCCL> pg,
    int M, int N) {
    
    const int warmup_iters = 100;
    const int measure_iters = 500;
    double total_time = 0.0;
    size_t mem_before = 0, mem_after = 0, mem_peak = 0;
    
    for (int iter = 0; iter < warmup_iters + measure_iters; ++iter) {
        // Start with column-sharded tensor
        DTensor X(mesh, pg);
        std::vector<int64_t> shape = {M, N};
        Layout col_sharded_layout(*mesh, shape, 1);
        
        // Each rank has N/world_size columns
        int world_size = mesh->size();
        std::vector<float> local_data(M * (N / world_size), 1.0f);
        X.setData(local_data, col_sharded_layout);
        
        size_t iter_mem_before = get_gpu_memory_used();
        
        // Measure transformation time
        auto start = Clock::now();
        X.replicate(0);  // AllGather to replicate
        cudaDeviceSynchronize();
        auto end = Clock::now();
        
        size_t iter_mem_after = get_gpu_memory_used();
        size_t iter_mem_peak = iter_mem_after;
        
        // Only record measurement iterations
        if (iter >= warmup_iters) {
            total_time += Duration(end - start).count();
            mem_before = iter_mem_before;
            mem_after = iter_mem_after;
            mem_peak = std::max(mem_peak, iter_mem_peak);
        }
    }
    
    double avg_time = total_time / measure_iters;
    
    // Data movement: AllGather exchanges (world_size-1)/world_size of total data per rank
    double data_moved_gb = (M * N * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gbps = data_moved_gb / (avg_time / 1000.0);
    
    LayoutBenchmarkResult result;
    result.transformation_name = "Sharded → Replicated";
    result.avg_time_ms = avg_time;
    result.data_moved_gb = data_moved_gb;
    result.bandwidth_gbps = bandwidth_gbps;
    result.memory_before_mb = mem_before;
    result.memory_after_mb = mem_after;
    result.memory_peak_mb = mem_peak;
    
    return result;
}

void print_result(int rank, const LayoutBenchmarkResult& result, int M, int N) {
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  " << std::setw(25) << std::left << result.transformation_name
                  << " | Time: " << std::setw(8) << std::right << result.avg_time_ms << " ms"
                  << " | Data: " << std::setw(6) << result.data_moved_gb << " GB"
                  << " | BW: " << std::setw(7) << result.bandwidth_gbps << " GB/s"
                  << " | Mem: " << result.memory_after_mb << " MB"
                  << std::endl;
    }
}

void run_benchmark_suite(int rank, std::shared_ptr<DeviceMesh> mesh,
                         std::shared_ptr<ProcessGroupNCCL> pg,
                         int M, int N) {
    if (rank == 0) {
        std::cout << "\n[Matrix size: " << M << "x" << N << "]" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run all transformations
    auto result1 = benchmark_replicate_to_row_shard(rank, mesh, pg, M, N);
    MPI_Barrier(MPI_COMM_WORLD);
    print_result(rank, result1, M, N);
    
    auto result2 = benchmark_replicate_to_col_shard(rank, mesh, pg, M, N);
    MPI_Barrier(MPI_COMM_WORLD);
    print_result(rank, result2, M, N);
    
    auto result3 = benchmark_row_to_col_shard(rank, mesh, pg, M, N);
    MPI_Barrier(MPI_COMM_WORLD);
    print_result(rank, result3, M, N);
    
    auto result4 = benchmark_shard_to_replicated(rank, mesh, pg, M, N);
    MPI_Barrier(MPI_COMM_WORLD);
    print_result(rank, result4, M, N);
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
    auto pg = init_process_group(world_size, rank);

    CUDA_CHECK(cudaSetDevice(rank));

    if (rank == 0) {
        std::cout << "\nLayout Transformations Benchmark\n";
        std::cout << "===================================\n";
        std::cout << "(100 warmup + 500 measurement iterations)\n";
    }

    // Run benchmarks with different matrix sizes
    run_benchmark_suite(rank, mesh, pg, 512, 512);
    run_benchmark_suite(rank, mesh, pg, 1024, 1024);
    run_benchmark_suite(rank, mesh, pg, 2048, 2048);
    run_benchmark_suite(rank, mesh, pg, 4096, 4096);
    run_benchmark_suite(rank, mesh, pg, 8192, 8192);

    if (rank == 0) {
        std::cout << "\nDone.\n";
    }

    MPI_Finalize();
    return 0;
}
