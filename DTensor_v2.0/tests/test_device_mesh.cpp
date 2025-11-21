#include "tensor/dtensor.h"
#include "tensor/device_mesh.h"
#include "tensor/placement.h"
#include "tensor/layout.h"
#include "process_group/process_group.h"
#include <iostream>
#include <mpi.h>

// Simple test for 1D DeviceMesh (backward compatibility test)
void test_1d_mesh() {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::cout << "[Rank " << rank << "] Testing 1D DeviceMesh...\n";
    
    // Create 1D DeviceMesh [2] for 2 GPUs
    auto device_mesh = std::make_shared<DeviceMesh>(
        std::vector<int>{world_size}
    );
    
    device_mesh->describe();
    
    // Create ProcessGroup (uses global communicator for now)
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    
    // Create DTensor with 1D sharding
    auto dtensor = std::make_shared<DTensor>(device_mesh, pg);
    
    // Create sharded layout: global shape [8, 16], sharded on dim 0
    std::vector<int> global_shape = {8, 16};
    std::vector<std::shared_ptr<Placement>> placements = {
        std::make_shared<Shard>(0)  // Shard along tensor dimension 0
    };
    Layout layout(device_mesh, global_shape, placements);
    
    std::cout << layout.describe(rank) << "\n";
    
    // Expected local shape for rank 0: [4, 16]
    // Expected local shape for rank 1: [4, 16]
    auto local_shape = layout.get_local_shape(rank);
    int expected_rows = global_shape[0] / world_size;
    
    if (local_shape[0] == expected_rows && local_shape[1] == global_shape[1]) {
        std::cout << "[Rank " << rank << "] ✓ 1D sharding test PASSED\n";
    } else {
        std::cout << "[Rank " << rank << "] ✗ 1D sharding test FAILED\n";
    }
}

// Test for replicated layout
void test_replicated() {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::cout << "[Rank " << rank << "] Testing Replicated layout...\n";
    
    auto device_mesh = std::make_shared<DeviceMesh>(
        std::vector<int>{world_size}
    );
    
    std::vector<int> global_shape = {10, 20};
    Layout layout = Layout::replicated(device_mesh, global_shape);
    
    std::cout << layout.describe(rank) << "\n";
    
    auto local_shape = layout.get_local_shape(rank);
    
    if (local_shape == global_shape) {
        std::cout << "[Rank " << rank << "] ✓ Replicated layout test PASSED\n";
    } else {
        std::cout << "[Rank " << rank << "] ✗ Replicated layout test FAILED\n";
    }
}

// Test for placement equality
void test_placement_equality() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        std::cout << "[Rank 0] Testing Placement equality...\n";
        
        auto shard1 = std::make_shared<Shard>(0);
        auto shard2 = std::make_shared<Shard>(0);
        auto shard3 = std::make_shared<Shard>(1);
        auto rep1 = std::make_shared<Replicate>();
        auto rep2 = std::make_shared<Replicate>();
        
        bool test1 = shard1->equals(shard2.get());  // Should be true
        bool test2 = shard1->equals(shard3.get());  // Should be false
        bool test3 = rep1->equals(rep2.get());      // Should be true
        bool test4 = shard1->equals(rep1.get());    // Should be false
        
        if (test1 && !test2 && test3 && !test4) {
            std::cout << "[Rank 0] ✓ Placement equality test PASSED\n";
        } else {
            std::cout << "[Rank 0] ✗ Placement equality test FAILED\n";
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "This test requires exactly 2 GPUs. Run with: mpirun -np 2 ./test_device_mesh\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        std::cout << "========================================\n";
        std::cout << "DeviceMesh & Placement Tests (2 GPUs)\n";
        std::cout << "========================================\n\n";
    }
    
    try {
        test_placement_equality();
        MPI_Barrier(MPI_COMM_WORLD);
        
        test_replicated();
        MPI_Barrier(MPI_COMM_WORLD);
        
        test_1d_mesh();
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "\n========================================\n";
            std::cout << "All tests completed!\n";
            std::cout << "========================================\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] Error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Finalize();
    return 0;
}
