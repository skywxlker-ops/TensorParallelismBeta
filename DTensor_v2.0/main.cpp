#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory> // For std::shared_ptr
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <stdexcept> // For std::runtime_error
#include <numeric>   // For std::iota

// === DTensor Core ===
#include "tensor/dtensor.h"
#include "process_group/process_group.h"
#include "tensor/mesh.h"
#include "tensor/layout.h"

// === TensorLib (for ops) ===
#include "bridge/tensor_ops_bridge.h"
#include "memory/cachingAllocator.hpp"

// === Utilities ===
// Note: planner.h is no longer needed
// #include "ckpt.h" // Checkpointing logic is commented out for this example

using namespace OwnTensor;

// =============================================================
// Test Case 1: Column-Parallel Matmul
// Y_col_shard = A_replicated @ B_col_shard
// =============================================================
void test_column_parallel_matmul(int rank, int world_size, 
                                 std::shared_ptr<Mesh> mesh, 
                                 std::shared_ptr<ProcessGroup> pg) {
    
    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << "\n"
                  << "TEST: Column-Parallel Matmul (Y_shard = A_repl @ B_col_shard)\n";
    }

    // --- 1. Define Global Shapes & Layouts ---
    int M = 4;
    int K = 8;
    int N = 8; 

    // A is [M, K] and Replicated
    std::vector<int> shape_A = {M, K};
    Layout layout_A(mesh, shape_A, ShardingType::REPLICATED);

    // B is [K, N] and Sharded on dim 1 (Column-Parallel)
    std::vector<int> shape_B = {K, N};
    Layout layout_B(mesh, shape_B, ShardingType::SHARDED, 1 /* shard_dim */);

    // --- 2. Create Local Data ---
    
    // A: All ranks get the full [M, K] tensor
    std::vector<int> local_shape_A = layout_A.get_local_shape(rank);
    int local_size_A = M * K;
    std::vector<float> data_A(local_size_A);
    std::iota(data_A.begin(), data_A.end(), 1.0f); // A = [1, 2, 3, ...]

    // B: Each rank gets its local shard, e.g., [K, N/world_size]
    std::vector<int> local_shape_B = layout_B.get_local_shape(rank);
    int local_size_B = local_shape_B[0] * local_shape_B[1];
    std::vector<float> data_B(local_size_B, 1.0f); // B_shard = [1, 1, 1, ...]

    // --- 3. Create and Set DTensors ---
    DTensor A(mesh, pg);
    A.setData(data_A, layout_A);

    DTensor B(mesh, pg);
    B.setData(data_B, layout_B);

    if (rank == 0) {
        std::cout << "\n--- Input A ---" << std::endl;
        A.print();
        std::cout << "\n--- Input B (Rank 0) ---" << std::endl;
        B.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
        std::cout << "\n--- Input B (Rank 1) ---" << std::endl;
        B.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- 4. Perform Distributed Matmul ---
    DTensor C = A.matmul(B);

    // --- 5. Print Output ---
    if (rank == 0) {
        std::cout << "\n--- Output C (Rank 0) ---" << std::endl;
        C.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
        std::cout << "\n--- Output C (Rank 1) ---" << std::endl;
        C.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


// =============================================================
// Test Case 2: Row-Parallel Matmul
// Y_replicated = A_row_shard @ B_replicated (performs AllGather)
// =============================================================
void test_row_parallel_matmul(int rank, int world_size, 
                              std::shared_ptr<Mesh> mesh, 
                              std::shared_ptr<ProcessGroup> pg) {
    
    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << "\n"
                  << "TEST: Row-Parallel Matmul (Y_repl = A_row_shard @ B_repl)\n";
    }

    // --- 1. Define Global Shapes & Layouts ---
    int M = 4;
    int K = 8;
    int N = 4; 

    // A is [M, K] and Sharded on dim 0 (Row-Parallel)
    std::vector<int> shape_A = {M, K};
    Layout layout_A(mesh, shape_A, ShardingType::SHARDED, 0 /* shard_dim */);

    // B is [K, N] and Replicated
    std::vector<int> shape_B = {K, N};
    Layout layout_B(mesh, shape_B, ShardingType::REPLICATED);

    // --- 2. Create Local Data ---
    
    // A: Each rank gets its local shard, e.g., [M/world_size, K]
    std::vector<int> local_shape_A = layout_A.get_local_shape(rank);
    int local_size_A = local_shape_A[0] * local_shape_A[1];
    // Fill with rank-specific data
    std::vector<float> data_A(local_size_A, (float)(rank + 1)); 

    // B: All ranks get the full [K, N] tensor
    std::vector<int> local_shape_B = layout_B.get_local_shape(rank);
    int local_size_B = K * N;
    std::vector<float> data_B(local_size_B, 1.0f); // B = [1, 1, 1, ...]

    // --- 3. Create and Set DTensors ---
    DTensor A(mesh, pg);
    A.setData(data_A, layout_A);

    DTensor B(mesh, pg);
    B.setData(data_B, layout_B);

    if (rank == 0) {
        std::cout << "\n--- Input A (Rank 0) ---" << std::endl;
        A.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
        std::cout << "\n--- Input A (Rank 1) ---" << std::endl;
        A.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- 4. Perform Distributed Matmul ---
    DTensor C = A.matmul(B);

    // --- 5. Print Output ---
    // The output C should be REPLICATED and identical on all ranks.
    if (rank == 0) {
        std::cout << "\n--- Output C (Rank 0) ---" << std::endl;
        C.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
        std::cout << "\n--- Output C (Rank 1) ---" << std::endl;
        C.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================
// Main Entry
// =============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
        std::cout << "[Init] Using DTensor (Layout-Aware) C++ Test\n";

    cudaFree(0); // Force CUDA context init

    // ------------------------------------------------------------
    // Initialize NCCL
    // ------------------------------------------------------------
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // Create NEW Mesh and ProcessGroup
    // ------------------------------------------------------------
    try {
        std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>(world_size);
        std::shared_ptr<ProcessGroup> pg = std::make_shared<ProcessGroup>(rank, world_size, rank, id);

        // --- Run Tests ---
        test_column_parallel_matmul(rank, world_size, mesh, pg);
        
        test_row_parallel_matmul(rank, world_size, mesh, pg);

    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] ERROR: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ------------------------------------------------------------
    // Finalize
    // ------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n=== Allocator Stats ===" << std::endl;
        gAllocator.printStats();
        std::cout << "\n[SUCCESS] All tests passed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}


