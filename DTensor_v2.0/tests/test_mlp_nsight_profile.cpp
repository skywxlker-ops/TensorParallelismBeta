#include <mpi.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

#include "tensor/dtensor.h"
#include "ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"

// =============================================================================
// MLP Profiling Test for NVIDIA Nsight
//
// Architecture (Tensor Parallel):
//   Layer 1 (Column-Parallel): X @ W1 -> GeLU -> H1
//   Layer 2 (Row-Parallel):    H1 @ W2 -> AllReduce -> Output
//
// Usage:
//   make test_mlp_nsight_profile
//   nsys profile -o mlp_profile mpirun -np 2 --allow-run-as-root ./test_mlp_nsight_profile
//   OR
//   ncu --set full -o mlp_kernel_profile mpirun -np 2 --allow-run-as-root ./test_mlp_nsight_profile
// =============================================================================

struct MLPConfig {
    int batch_size;
    int hidden_dim;
    int intermediate_dim;
    int num_iterations;
    
    void print(int rank) const {
        if (rank == 0) {
            std::cout << "\n=== MLP Configuration ===\n";
            std::cout << "  Batch Size:       " << batch_size << "\n";
            std::cout << "  Hidden Dim:       " << hidden_dim << "\n";
            std::cout << "  Intermediate Dim: " << intermediate_dim << "\n";
            std::cout << "  Iterations:       " << num_iterations << "\n";
        }
    }
};

class TensorParallelMLP {
public:
    TensorParallelMLP(const MLPConfig& config,
                      std::shared_ptr<DeviceMesh> mesh,
                      std::shared_ptr<ProcessGroupNCCL> pg,
                      int rank)
        : config_(config), mesh_(mesh), pg_(pg), rank_(rank) {
        
        init_weights();
    }
    
    void init_weights() {
        int world_size = mesh_->world_size();
        
        // W1: [hidden, intermediate] - column sharded
        std::vector<int64_t> shape_W1 = {config_.hidden_dim, config_.intermediate_dim};
        Layout layout_W1(mesh_, shape_W1, 1);
        auto local_W1 = layout_W1.get_local_shape(rank_);
        int size_W1 = local_W1[0] * local_W1[1];
        std::vector<float> data_W1(size_W1, 0.01f);
        W1_ = std::make_unique<DTensor>(mesh_, pg_);
        W1_->setData(data_W1, layout_W1);
        
        // W2: [intermediate, hidden] - row sharded
        std::vector<int64_t> shape_W2 = {config_.intermediate_dim, config_.hidden_dim};
        Layout layout_W2(mesh_, shape_W2, 0);
        auto local_W2 = layout_W2.get_local_shape(rank_);
        int size_W2 = local_W2[0] * local_W2[1];
        std::vector<float> data_W2(size_W2, 0.01f);
        W2_ = std::make_unique<DTensor>(mesh_, pg_);
        W2_->setData(data_W2, layout_W2);
    }
    
    DTensor forward(DTensor& input) {
        // Layer 1: Column-parallel matmul
        DTensor h1 = input.matmul(*W1_);
        
        // Layer 2: Row-parallel matmul (includes AllReduce)
        DTensor output = h1.matmul(*W2_);
        
        return output;
    }
    
private:
    MLPConfig config_;
    std::shared_ptr<DeviceMesh> mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    int rank_;
    
    std::unique_ptr<DTensor> W1_;
    std::unique_ptr<DTensor> W2_;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    cudaSetDevice(rank);
    
    // Configuration (adjust for your profiling needs)
    MLPConfig config;
    config.batch_size = 32;
    config.hidden_dim = 4096;
    config.intermediate_dim = 4 * 4096;  // 4x hidden (GPT-style)
    config.num_iterations = 100;
    
    config.print(rank);
    
    if (rank == 0) {
        std::cout << "  World Size:       " << world_size << "\n";
        std::cout << "  Shard/Rank:       " << config.intermediate_dim / world_size << "\n";
    }
    
    // Initialize process group and mesh
    auto pg = init_process_group(world_size, rank);
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    
    // Create MLP
    TensorParallelMLP mlp(config, mesh, pg, rank);
    
    // Create input tensor (replicated)
    std::vector<int64_t> shape_X = {config.batch_size, config.hidden_dim};
    Layout layout_X = Layout::replicated(*mesh, shape_X);
    std::vector<float> data_X(config.batch_size * config.hidden_dim, 1.0f);
    DTensor input(mesh, pg);
    input.setData(data_X, layout_X);
    
    // Warmup iterations (not profiled)
    if (rank == 0) std::cout << "\n=== Warmup (5 iterations) ===\n";
    for (int i = 0; i < 5; i++) {
        DTensor output = mlp.forward(input);
        cudaDeviceSynchronize();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Start profiling
    if (rank == 0) std::cout << "\n=== Starting Profiled Region ===\n";
    cudaProfilerStart();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Main profiling loop
    for (int i = 0; i < config.num_iterations; i++) {
        DTensor output = mlp.forward(input);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Stop profiling
    cudaProfilerStop();
    
    // Calculate timing
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_iter_ms = elapsed_ms / config.num_iterations;
    
    // Calculate FLOPS (2 * M * N * K for matmul)
    // Layer1: batch * hidden * inter
    // Layer2: batch * inter * hidden
    int64_t flops_per_iter = 
        2LL * config.batch_size * config.hidden_dim * config.intermediate_dim +   // W1 matmul
        2LL * config.batch_size * config.intermediate_dim * config.hidden_dim;    // W2 matmul
    
    double tflops = (flops_per_iter * config.num_iterations) / (elapsed_ms * 1e9);
    
    if (rank == 0) {
        std::cout << "\n=== Results ===\n";
        std::cout << "  Total time:     " << elapsed_ms << " ms\n";
        std::cout << "  Avg per iter:   " << avg_iter_ms << " ms\n";
        std::cout << "  Throughput:     " << tflops << " TFLOPS\n";
        std::cout << "\n=== Profiling Complete ===\n";
        std::cout << "Use nsys/ncu to analyze the profile.\n\n";
    }
    
    MPI_Finalize();
    return 0;
}

// =============================================================================
// COMMANDS REFERENCE
// =============================================================================
//
// 1. BUILD:
//    cd tests
//    make test_mlp_nsight_profile
//
// 2. RUN (verify it works):
//    mpirun -np 2 --allow-run-as-root ./test_mlp_nsight_profile
//
// 3. PROFILE WITH NSIGHT SYSTEMS (timeline view):
//    nsys profile -o mlp_profile --force-overwrite true mpirun -np 2 --allow-run-as-root ./test_mlp_nsight_profile
//
// 4. VIEW NSIGHT SYSTEMS REPORT:
//    nsys-ui mlp_profile.nsys-rep          # Open in GUI
//    nsys stats mlp_profile.nsys-rep       # Print stats to terminal
//
// 5. PROFILE WITH NSIGHT COMPUTE (kernel-level metrics):
//    ncu --set full -o mlp_kernel mpirun -np 2 --allow-run-as-root ./test_mlp_nsight_profile
//
// 6. VIEW NSIGHT COMPUTE REPORT:
//    ncu-ui mlp_kernel.ncu-rep             # Open in GUI
//
// =============================================================================

