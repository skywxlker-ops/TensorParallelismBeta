#include "tensor/dtensor.h"
#include "tensor/device_mesh.h"
#include "tensor/placement.h"
#include "tensor/layout.h"
#include "process_group/process_group.h"
#include <iostream>
#include <mpi.h>


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

void test_1d_mesh(int rank, int world_size) {
    std::cout << "[Rank " << rank << "] Testing 1D DeviceMesh...\n";
    
 
    auto device_mesh = std::make_shared<DeviceMesh>(
        std::vector<int>{world_size}
    );
    
    device_mesh->describe();
    
   
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    

    auto dtensor = std::make_shared<DTensor>(device_mesh, pg);
    

    std::vector<int> global_shape = {8, 16};
    std::vector<std::shared_ptr<Placement>> placements = {
        std::make_shared<Shard>(0) 
    };
    Layout layout(device_mesh, global_shape, placements);
    
    std::cout << layout.describe(rank) << "\n";
    
    auto local_shape = layout.get_local_shape(rank);
    int expected_rows = global_shape[0] / world_size;
    
    if (local_shape[0] == expected_rows && local_shape[1] == global_shape[1]) {
        std::cout << "[Rank " << rank << "]   1D sharding test PASSED\n";
    } else {
        std::cout << "[Rank " << rank << "]   1D sharding test FAILED\n";
    }
}

// Test for 2D DeviceMesh 
void test_2d_mesh(int rank, int world_size) {
    if (world_size < 4) {
        if (rank == 0) std::cout << "Skipping 2D mesh test (needs >= 4 GPUs for non-degenerate mesh)\n";
        return;
    }

    std::vector<int> mesh_shape;
    if (world_size == 4) {
        mesh_shape = {2, 2};  
    } else if (world_size == 8) {
        mesh_shape = {2, 4};  
    } else if (world_size == 16) {
        mesh_shape = {4, 4};  
    } else {
    
        int dim0 = static_cast<int>(sqrt(world_size));
        while (world_size % dim0 != 0 && dim0 > 1) dim0--;
        mesh_shape = {dim0, world_size / dim0};
    }

    if (rank == 0) {
        std::cout << "Testing 2D DeviceMesh [" << mesh_shape[0] << ", " << mesh_shape[1] << "]...\n";
    }
    

    auto device_mesh = std::make_shared<DeviceMesh>(mesh_shape);
    

    std::vector<int> coords = device_mesh->get_coordinate(rank);
    
    std::cout << "[DeviceMesh 2D] Rank " << rank << "/" << world_size 
              << " | Shape: [" << mesh_shape[0] << ", " << mesh_shape[1] << "]"
              << " | Coordinate: [" << coords[0] << ", " << coords[1] << "]\n";
              
 
    int expected_coord0 = rank / mesh_shape[1];
    int expected_coord1 = rank % mesh_shape[1];
    
    if (coords[0] != expected_coord0 || coords[1] != expected_coord1) {
        std::cerr << "[Rank " << rank << "]  2D Coordinate mismatch! Expected ["
                  << expected_coord0 << ", " << expected_coord1 << "]\n";
        exit(1);
    }
    

    std::vector<int> global_shape = {mesh_shape[1] * 10, 20};  
    std::vector<std::shared_ptr<Placement>> placements = {
        std::make_shared<Replicate>(), 
        std::make_shared<Shard>(0)     
    };
    
    Layout layout(device_mesh, global_shape, placements);
    std::vector<int> local_shape = layout.get_local_shape(rank);
    
   
    int expected_dim0 = global_shape[0] / mesh_shape[1];
    int expected_dim1 = global_shape[1]; 
    
    std::cout << "[Layout 2D] Rank " << rank << "/" << world_size
              << " | Global: [" << global_shape[0] << ", " << global_shape[1] << "]"
              << " | Placements: [Replicate, Shard(0)]"
              << " | Local: [" << local_shape[0] << ", " << local_shape[1] << "]\n";
              
    if (local_shape[0] != expected_dim0 || local_shape[1] != expected_dim1) {
        std::cerr << "[Rank " << rank << "]  2D Layout mismatch! Expected ["
                  << expected_dim0 << ", " << expected_dim1 << "]\n";
        exit(1);
    }

    if (rank == 0) std::cout << " 2D mesh test PASSED\n";
}

void test_3d_mesh(int rank, int world_size) {
    if (world_size < 8) {
        if (rank == 0) std::cout << "Skipping 3D mesh test (needs >= 8 GPUs for non-degenerate mesh)\n";
        return;
    }

    std::vector<int> mesh_shape;
    if (world_size == 8) {
        mesh_shape = {2, 2, 2};  
    } else if (world_size == 16) {
        mesh_shape = {2, 2, 4};  
    } else if (world_size == 32) {
        mesh_shape = {2, 4, 4};  
    } else {
    
        int dim0 = static_cast<int>(sqrt(world_size));
        while (world_size % dim0 != 0 && dim0 > 1) dim0--;
        mesh_shape = {dim0, world_size / dim0};
    }

    if (rank == 0) {
        std::cout << "Testing 3D DeviceMesh [" << mesh_shape[0] << ", " << mesh_shape[1] << ", " << mesh_shape[2] << "]...\n";
    }
    

    auto device_mesh = std::make_shared<DeviceMesh>(mesh_shape);
    

    std::vector<int> coords = device_mesh->get_coordinate(rank);
    
    std::cout << "[DeviceMesh 3D] Rank " << rank << "/" << world_size 
              << " | Shape: [" << mesh_shape[0] << ", " << mesh_shape[1] << ", " << mesh_shape[2] << "]"
              << " | Coordinate: [" << coords[0] << ", " << coords[1] << ", " << coords[2] << "]\n";
              
 
    int expected_coord0 = rank / mesh_shape[1];
    int expected_coord1 = rank % mesh_shape[1];
    int expected_coord2 = rank % mesh_shape[2];
    
    if (coords[0] != expected_coord0 || coords[1] != expected_coord1 || coords[2] != expected_coord2) {
        std::cerr << "[Rank " << rank << "]  3D Coordinate mismatch! Expected ["
                  << expected_coord0 << ", " << expected_coord1 << ", " << expected_coord2 << "]\n";
        exit(1);
    }
    

    std::vector<int> global_shape = {mesh_shape[0] * 10, mesh_shape[1] * 20, mesh_shape[2] * 30};  
    std::vector<std::shared_ptr<Placement>> placements = {
        std::make_shared<Replicate>(), 
        std::make_shared<Shard>(0)     
    };
    
    Layout layout(device_mesh, global_shape, placements);
    std::vector<int> local_shape = layout.get_local_shape(rank);
    
   
    int expected_dim0 = global_shape[0] / mesh_shape[0];
    int expected_dim1 = global_shape[1]; 
    int expected_dim2 = global_shape[2]; 
    
    std::cout << "[Layout 3D] Rank " << rank << "/" << world_size
              << " | Global: [" << global_shape[0] << ", " << global_shape[1] << ", " << global_shape[2] << "]"
              << " | Placements: [Replicate, Shard(0)]"
              << " | Local: [" << local_shape[0] << ", " << local_shape[1] << ", " << local_shape[2] << "]\n";
              
    if (local_shape[0] != expected_dim0 || local_shape[1] != expected_dim1 || local_shape[2] != expected_dim2) {
        std::cerr << "[Rank " << rank << "]  3D Layout mismatch! Expected ["
                  << expected_dim0 << ", " << expected_dim1 << ", " << expected_dim2 << "]\n";
        exit(1);
    }

    if (rank == 0) std::cout << " 3D mesh test PASSED\n";
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (rank == 0) {

        std::cout << "DeviceMesh & Placement Tests (" << world_size << " GPUs)\n";

    }
    
    try {
        test_placement_equality(rank);
        MPI_Barrier(MPI_COMM_WORLD);
        
        test_replicated_layout(rank, world_size);
        MPI_Barrier(MPI_COMM_WORLD);
        
        test_1d_mesh(rank, world_size);
        MPI_Barrier(MPI_COMM_WORLD);
        
        test_2d_mesh(rank, world_size);
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
 
            std::cout << "All tests completed\n";

        }
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] Error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Finalize();
    return 0;
}
