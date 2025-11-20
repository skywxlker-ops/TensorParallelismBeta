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

using namespace OwnTensor;

// =============================================================
// Full Tensor Parallel MLP Test
// Layer 1 (Column): X_repl @ W1_col_shard -> Y1_col_shard
// Layer 2 (Row):    Y1_col_shard @ W2_row_shard -> Y2_repl
// =============================================================
void test_tensor_parallel_mlp(int rank, int world_size, 
                              std::shared_ptr<Mesh> mesh, 
                              std::shared_ptr<ProcessGroup> pg) {
    
    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << "\n"
                  << "TEST: Full Tensor Parallel MLP (Col Parallel -> Row Parallel)\n";
    }

    int BATCH = 2;
    int HIDDEN = 4;
    int INTERMEDIATE = 8; // 4 * HIDDEN / WorldSize (assuming 2) -> Total 8

    // ---------------------------------------------------------
    // Step 1: Input X (Replicated)
    // ---------------------------------------------------------
    std::vector<int> shape_X = {BATCH, HIDDEN};
    Layout layout_X(mesh, shape_X, ShardingType::REPLICATED);
    
    // X = All 1s
    std::vector<float> data_X(BATCH * HIDDEN, 1.0f); 
    DTensor X(mesh, pg);
    X.setData(data_X, layout_X);

    if (rank == 0) {
        std::cout << "\n--- [1] Input X (Replicated) ---" << std::endl;
        X.print();
    }

    // ---------------------------------------------------------
    // Step 2: Layer 1 Weights W1 (Column Sharded)
    // Global: [HIDDEN, 8], Local: [HIDDEN, 4] (if world_size=2)
    // ---------------------------------------------------------
    std::vector<int> shape_W1 = {HIDDEN, 8};
    Layout layout_W1(mesh, shape_W1, ShardingType::SHARDED, 1 /* col dim */);

    std::vector<int> local_shape_W1 = layout_W1.get_local_shape(rank);
    int size_W1 = local_shape_W1[0] * local_shape_W1[1];
    
    // Rank 0 gets 0.5s, Rank 1 gets 1.0s
    std::vector<float> data_W1(size_W1, (rank + 1) * 0.5f); 
    DTensor W1(mesh, pg);
    W1.setData(data_W1, layout_W1);

    // ---------------------------------------------------------
    // Step 3: Column Parallel MatMul (X @ W1)
    // ---------------------------------------------------------
    DTensor Y1 = X.matmul(W1);

    if (rank == 0) std::cout << "\n--- [2] Y1 Output (Col Sharded) ---" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) Y1.print();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) Y1.print();
    MPI_Barrier(MPI_COMM_WORLD);

    // ---------------------------------------------------------
    // Step 4: Layer 2 Weights W2 (Row Sharded)
    // Global: [8, HIDDEN], Local: [4, HIDDEN] (if world_size=2)
    // Sharded on Dim 0 (Rows) to accept the Col-Sharded input Y1
    // ---------------------------------------------------------
    std::vector<int> shape_W2 = {8, HIDDEN};
    Layout layout_W2(mesh, shape_W2, ShardingType::SHARDED, 0 /* row dim */);

    std::vector<int> local_shape_W2 = layout_W2.get_local_shape(rank);
    int size_W2 = local_shape_W2[0] * local_shape_W2[1];
    
    // Rank 0 gets 1.0s, Rank 1 gets 1.0s
    std::vector<float> data_W2(size_W2, 1.0f); 
    DTensor W2(mesh, pg);
    W2.setData(data_W2, layout_W2);

    // ---------------------------------------------------------
    // Step 5: Row Parallel MatMul (Y1 @ W2)
    // Splits dot product (K dim) -> Partial Sum -> AllReduce
    // ---------------------------------------------------------
    DTensor Y2 = Y1.matmul(W2);

    if (rank == 0) std::cout << "\n--- [3] Y2 Output (Replicated via AllReduce) ---" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Both ranks should print the exact same result
    if (rank == 0) Y2.print();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) Y2.print();
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
        std::cout << "[Init] DTensor v2.0 Test Driver\n";

    cudaFree(0); // Force CUDA context init

    // Initialize NCCL
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    try {
        std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>(world_size);
        // Simple round-robin device assignment
        std::shared_ptr<ProcessGroup> pg = std::make_shared<ProcessGroup>(rank, world_size, rank % 4, id);

        // Run the Full TP Test
        test_tensor_parallel_mlp(rank, world_size, mesh, pg);

    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] ERROR: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n=== Allocator Stats ===" << std::endl;
        gAllocator.printStats();
        std::cout << "\n[SUCCESS] All tests passed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}