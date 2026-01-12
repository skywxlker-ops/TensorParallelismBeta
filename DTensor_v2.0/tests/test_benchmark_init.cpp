#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>

#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include "bridge/bridge.h"

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
    size_t memory_used_mb;
    
    void print(const std::string& method, int rank) const {
        if (rank == 0) {
            std::cout << "  " << std::setw(20) << std::left << method 
                      << " | " << std::setw(10) << std::fixed << std::setprecision(3) 
                      << init_time_ms << " ms"
                      << " | " << memory_used_mb << " MB" << std::endl;
        }
    }
};

size_t get_gpu_memory_used() {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return total - free;
}

BenchmarkResult benchmark_old_method(int rank, std::shared_ptr<DeviceMesh> mesh,
                                      std::shared_ptr<ProcessGroupNCCL> pg,
                                      const Layout& layout,
                                      const std::vector<float>& local_data) {
    size_t mem_before = get_gpu_memory_used();
    
    auto start = Clock::now();
    
    DTensor tensor(mesh, pg);
    tensor.setData(local_data, layout);
    
    auto end = Clock::now();
    
    size_t mem_after = get_gpu_memory_used();
    
    return {
        Duration(end - start).count(),
        (mem_after - mem_before) / (1024 * 1024)
    };
}

BenchmarkResult benchmark_new_method(int rank, std::shared_ptr<DeviceMesh> mesh,
                                      std::shared_ptr<ProcessGroupNCCL> pg,
                                      const Layout& layout,
                                      const std::vector<float>& full_data) {
    size_t mem_before = get_gpu_memory_used();
    
    auto start = Clock::now();
    
    DTensor tensor(mesh, pg);
    tensor.setDataFromRoot(full_data, layout, 0);
    
    auto end = Clock::now();
    
    size_t mem_after = get_gpu_memory_used();
    
    return {
        Duration(end - start).count(),
        (mem_after - mem_before) / (1024 * 1024)
    };
}

void run_benchmark(int rank, int world_size,
                   std::shared_ptr<DeviceMesh> mesh,
                   std::shared_ptr<ProcessGroupNCCL> pg,
                   int rows, int cols) {
    
    if (rank == 0) {
        int size_mb = (rows * cols * 4) / (1024 * 1024);
        std::cout << "\n[" << rows << "x" << cols << " = " << size_mb << "MB]" << std::endl;
    }
    
    std::vector<int64_t> global_shape = {rows, cols};
    Layout layout(*mesh, global_shape, 0);
    
    int local_rows = rows / world_size;
    int local_size = local_rows * cols;
    int global_size = rows * cols;
    
    // Prepare data for old method (each rank has its own chunk)
    std::vector<float> local_data(local_size);
    for (int i = 0; i < local_size; ++i) {
        local_data[i] = static_cast<float>(rank * local_size + i);
    }
    
    // Prepare data for new method (only root has full data)
    std::vector<float> full_data;
    if (rank == 0) {
        full_data.resize(global_size);
        for (int i = 0; i < global_size; ++i) {
            full_data[i] = static_cast<float>(i);
        }
    }
    
    // Benchmark old method
    auto old_result = benchmark_old_method(rank, mesh, pg, layout, local_data);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Benchmark new method
    auto new_result = benchmark_new_method(rank, mesh, pg, layout, full_data);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Print results
    if (rank == 0) {
        std::cout << "  Old: " << std::fixed << std::setprecision(2) 
                  << old_result.init_time_ms << "ms";
        std::cout << " | New: " << new_result.init_time_ms << "ms";
        
        double speedup = old_result.init_time_ms / new_result.init_time_ms;
        if (speedup < 1.0) {
            std::cout << " (" << std::setprecision(1) << (1.0/speedup) << "x slower)";
        } else {
            std::cout << " (" << std::setprecision(1) << speedup << "x faster)";
        }
        std::cout << std::endl;
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

    ncclUniqueId nccl_id;
    if (rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }
    MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    auto pg = init_process_group(world_size, rank);

    CUDA_CHECK(cudaSetDevice(rank));

    if (rank == 0) {
        std::cout << "\nBenchmark: setData() vs setDataFromRoot()\n";
    }

    // Run benchmarks with different sizes
    run_benchmark(rank, world_size, mesh, pg, 1024, 1024);    // 4 MB
    run_benchmark(rank, world_size, mesh, pg, 2048, 2048);    // 16 MB
    run_benchmark(rank, world_size, mesh, pg, 4096, 4096);    // 64 MB
    run_benchmark(rank, world_size, mesh, pg, 8192, 8192);    // 256 MB

    if (rank == 0) {
        std::cout << "\nDone.\n";
    }

    MPI_Finalize();
    return 0;
}
