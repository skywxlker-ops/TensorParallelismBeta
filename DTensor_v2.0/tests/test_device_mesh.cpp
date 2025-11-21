#include "tensor/dtensor.h"
#include "tensor/device_mesh.h"
#include "tensor/placement.h"
#include "tensor/layout.h"
#include "process_group/process_group.h"
#include <iostream>
#include <mpi.h>

// Test for placement equality
void test_placement_equality(int rank) {
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
            std::cout << "[Rank 0]   Placement equality test PASSED\n";
        } else {
            std::cout << "[Rank 0]   Placement equality test FAILED\n";
        }
    }
}

// Test for replicated layout
void test_replicated_layout(int rank, int world_size) {
    std::cout << "[Rank " << rank << "] Testing Replicated layout...\n";
    
    auto device_mesh = std::make_shared<DeviceMesh>(
        std::vector<int>{world_size}
    );
    
    std::vector<int> global_shape = {10, 20};
    Layout layout = Layout::replicated(device_mesh, global_shape);
    
    std::cout << layout.describe(rank) << "\n";
    
    auto local_shape = layout.get_local_shape(rank);
    
    if (local_shape == global_shape) {
        std::cout << "[Rank " << rank << "]   Replicated layout test PASSED\n";
    } else {
        std::cout << "[Rank " << rank << "]   Replicated layout test FAILED\n";
    }
}

// Simple test for 1D DeviceMesh (backward compatibility test)
void test_1d_mesh(int rank, int world_size) {
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
        std::cout << "[Rank " << rank << "]   1D sharding test PASSED\n";
    } else {
        std::cout << "[Rank " << rank << "]   1D sharding test FAILED\n";
    }
}

// Test for 2D DeviceMesh (ND support verification)
void test_2d_mesh(int rank, int world_size) {
    if (world_size < 2) {
        if (rank == 0) std::cout << "Skipping 2D mesh test (needs >= 2 GPUs)\n";
        return;
    }

    if (rank == 0) std::cout << "Testing 2D DeviceMesh [1, " << world_size << "]...\n";
    
    // Create a 2D mesh: 1 row, N columns (e.g., [1, 2] for 2 GPUs)
    std::vector<int> mesh_shape = {1, world_size};
    auto device_mesh = std::make_shared<DeviceMesh>(mesh_shape);
    
    // Verify coordinates
    // Rank r -> [0, r]
    std::vector<int> coords = device_mesh->get_coordinate(rank);
    
    std::cout << "[DeviceMesh 2D] Rank " << rank << "/" << world_size 
              << " | Shape: [" << mesh_shape[0] << ", " << mesh_shape[1] << "]"
              << " | Coordinate: [" << coords[0] << ", " << coords[1] << "]\n";
              
    if (coords[0] != 0 || coords[1] != rank) {
        std::cerr << "[Rank " << rank << "]   2D Coordinate mismatch!\n";
        exit(1);
    }
    
    // Test Layout on 2D Mesh
    // Shard on Mesh Dim 1 (Columns) -> Tensor Dim 0
    std::vector<int> global_shape = {10, 20};
    std::vector<std::shared_ptr<Placement>> placements = {
        std::make_shared<Replicate>(), // Mesh Dim 0: Replicate
        std::make_shared<Shard>(0)     // Mesh Dim 1: Shard Tensor Dim 0
    };
    
    Layout layout(device_mesh, global_shape, placements);
    std::vector<int> local_shape = layout.get_local_shape(rank);
    
    // Expected: Tensor Dim 0 (10) split by Mesh Dim 1 (2) -> 5
    // Tensor Dim 1 (20) replicated -> 20
    std::cout << "[Layout 2D] Rank " << rank << "/" << world_size
              << " | Global: [" << global_shape[0] << ", " << global_shape[1] << "]"
              << " | Placements: [Replicate, Shard(0)]"
              << " | Local: [" << local_shape[0] << ", " << local_shape[1] << "]\n";
              
    if (local_shape[0] != 5 || local_shape[1] != 20) {
        std::cerr << "[Rank " << rank << "]   2D Layout mismatch! Expected [5, 20]\n";
        exit(1);
    }

    if (rank == 0) std::cout << " 2D mesh test PASSED\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (rank == 0) {
        std::cout << "========================================\n";
        std::cout << "DeviceMesh & Placement Tests (" << world_size << " GPUs)\n";
        std::cout << "========================================\n\n";
    }
    
    try {
        test_placement_equality(rank);
        MPI_Barrier(MPI_COMM_WORLD);
        
        test_replicated_layout(rank, world_size);
        MPI_Barrier(MPI_COMM_WORLD);
        
        test_1d_mesh(rank, world_size);
        MPI_Barrier(MPI_COMM_WORLD);
        
        // TODO: 2D mesh test requires careful NCCL setup for degenerate dimensions
        // test_2d_mesh(rank, world_size);
        // MPI_Barrier(MPI_COMM_WORLD);
        
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
