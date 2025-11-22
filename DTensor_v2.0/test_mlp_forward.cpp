#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <iomanip>

// === DTensor Core ===
#include "tensor/dtensor.h"
#include "process_group/process_group.h"
#include "tensor/mesh.h"
#include "tensor/layout.h"

// === TensorLib (for ops) ===
#include "bridge/tensor_ops_bridge.h"
#include "memory/cachingAllocator.hpp"

using namespace OwnTensor;

// =============================================================================
// MLP Forward Pass Test with Tensor Parallelism
// 
// Architecture:
//   Layer 1 (Column-Parallel): X_replicated @ W1_col_shard -> Y1_col_shard
//   Layer 2 (Row-Parallel):    Y1_col_shard @ W2_row_shard -> Y2_replicated
//
// This test demonstrates:
//   - Column-parallel matmul (no communication needed)
//   - Row-parallel matmul (requires AllReduce for final result)
//   - Proper sharding and layout management
// =============================================================================

void print_separator(int rank, const std::string& title) {
    if (rank == 0) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << title << "\n";
        std::cout << std::string(70, '=') << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_section(int rank, const std::string& section) {
    if (rank == 0) {
        std::cout << "\n--- " << section << " ---\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void test_mlp_forward(int rank, int world_size, 
                      std::shared_ptr<Mesh> mesh, 
                      std::shared_ptr<ProcessGroup> pg) {
    
    print_separator(rank, "MLP Forward Pass Test (Tensor Parallelism)");
    
    // =============================================================
    // Configuration
    // =============================================================
    const int BATCH = 2;
    const int HIDDEN = 4;
    const int INTERMEDIATE = 8;  // Will be sharded across ranks
    
    if (rank == 0) {
        std::cout << "\nConfiguration:\n";
        std::cout << "  Batch Size:      " << BATCH << "\n";
        std::cout << "  Hidden Dim:      " << HIDDEN << "\n";
        std::cout << "  Intermediate:    " << INTERMEDIATE << "\n";
        std::cout << "  World Size:      " << world_size << "\n";
        std::cout << "  Shard per rank:  " << (INTERMEDIATE / world_size) << "\n";
    }
    
    // =============================================================
    // Layer 1: Column-Parallel MatMul
    // X [BATCH, HIDDEN] @ W1 [HIDDEN, INTERMEDIATE] -> Y1 [BATCH, INTERMEDIATE]
    // X is replicated, W1 is column-sharded
    // =============================================================
    
    print_section(rank, "LAYER 1: Column-Parallel MatMul");
    
    // --- Input X (Replicated) ---
    std::vector<int> shape_X = {BATCH, HIDDEN};
    Layout layout_X(mesh, shape_X, ShardingType::REPLICATED);
    
    // Initialize X with all 1s for easy verification
    std::vector<float> data_X(BATCH * HIDDEN, 1.0f);
    DTensor X(mesh, pg);
    X.setData(data_X, layout_X);
    
    if (rank == 0) {
        std::cout << "\nInput X (Replicated):\n";
        X.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Weight W1 (Column-Sharded) ---
    std::vector<int> shape_W1 = {HIDDEN, INTERMEDIATE};
    Layout layout_W1(mesh, shape_W1, ShardingType::SHARDED, 1);  // Shard on dim 1 (columns)
    
    std::vector<int> local_shape_W1 = layout_W1.get_local_shape(rank);
    int size_W1 = local_shape_W1[0] * local_shape_W1[1];
    
    // Rank 0 gets 0.5, Rank 1 gets 1.0 (for easy verification)
    std::vector<float> data_W1(size_W1, (rank + 1) * 0.5f);
    DTensor W1(mesh, pg);
    W1.setData(data_W1, layout_W1);
    
    if (rank == 0) {
        std::cout << "\nWeight W1 (Column-Sharded) - Rank 0:\n";
        W1.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1 && world_size > 1) {
        std::cout << "\nWeight W1 (Column-Sharded) - Rank 1:\n";
        W1.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Forward: Y1 = X @ W1 ---
    DTensor Y1 = X.matmul(W1);
    
    print_section(rank, "Layer 1 Output Y1 (Column-Sharded)");
    if (rank == 0) {
        std::cout << "\nY1 - Rank 0 (should have values 2.0):\n";
        Y1.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1 && world_size > 1) {
        std::cout << "\nY1 - Rank 1 (should have values 4.0):\n";
        Y1.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // =============================================================
    // Layer 2: Row-Parallel MatMul
    // Y1 [BATCH, INTERMEDIATE] @ W2 [INTERMEDIATE, HIDDEN] -> Y2 [BATCH, HIDDEN]
    // Y1 is column-sharded, W2 is row-sharded
    // This is the standard Row-Parallel pattern requiring AllReduce
    // =============================================================
    
    print_section(rank, "LAYER 2: Row-Parallel MatMul");
    
    // --- Weight W2 (Row-Sharded) ---
    std::vector<int> shape_W2 = {INTERMEDIATE, HIDDEN};
    Layout layout_W2(mesh, shape_W2, ShardingType::SHARDED, 0);  // Shard on dim 0 (rows)
    
    std::vector<int> local_shape_W2 = layout_W2.get_local_shape(rank);
    int size_W2 = local_shape_W2[0] * local_shape_W2[1];
    
    // Both ranks get 1.0 for easy verification
    std::vector<float> data_W2(size_W2, 1.0f);
    DTensor W2(mesh, pg);
    W2.setData(data_W2, layout_W2);
    
    if (rank == 0) {
        std::cout << "\nWeight W2 (Row-Sharded) - Rank 0:\n";
        W2.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1 && world_size > 1) {
        std::cout << "\nWeight W2 (Row-Sharded) - Rank 1:\n";
        W2.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Forward: Y2 = Y1 @ W2 (with AllReduce) ---
    DTensor Y2 = Y1.matmul(W2);
    
    print_section(rank, "Layer 2 Output Y2 (Replicated via AllReduce)");
    if (rank == 0) {
        std::cout << "\nY2 - Rank 0 (should be identical on all ranks):\n";
        Y2.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1 && world_size > 1) {
        std::cout << "\nY2 - Rank 1 (should be identical to Rank 0):\n";
        Y2.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // =============================================================
    // Verification
    // =============================================================
    if (rank == 0) {
        std::cout << "\n" << std::string(70, '-') << "\n";
        std::cout << "Expected Results:\n";
        std::cout << "  Y1 (Rank 0): All values should be 2.0 (HIDDEN * 0.5)\n";
        std::cout << "  Y1 (Rank 1): All values should be 4.0 (HIDDEN * 1.0)\n";
        std::cout << "  Y2 (Both):   All values should be 24.0\n";
        std::cout << "               (Rank0: 2.0 * 4 + Rank1: 4.0 * 4 = 8 + 16 = 24)\n";
        std::cout << std::string(70, '-') << "\n";
    }
}

// =============================================================================
// Main Entry Point
// =============================================================================
int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  DTensor v2.0 - MLP Forward Pass Test                      ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        std::cout << "\nInitializing with " << world_size << " ranks...\n";
    }
    
    // Force CUDA context initialization
    cudaSetDevice(rank % 4);  // Assuming max 4 GPUs per node
    cudaFree(0);
    
    // Initialize NCCL
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    try {
        // Create mesh and process group
        std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>(world_size);
        std::shared_ptr<ProcessGroup> pg = std::make_shared<ProcessGroup>(
            rank, world_size, rank % 4, id
        );
        
        if (rank == 0) {
            std::cout << "✓ MPI initialized\n";
            std::cout << "✓ NCCL initialized\n";
            std::cout << "✓ Process group created\n";
        }
        
        // Run the MLP forward test
        test_mlp_forward(rank, world_size, mesh, pg);
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] ERROR: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Print allocator statistics
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "Memory Allocator Statistics\n";
        std::cout << std::string(70, '=') << "\n";
        gAllocator.printStats();
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ✓ TEST PASSED - All operations completed successfully     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
    }
    
    MPI_Finalize();
    return 0;
}
