#include <mpi.h>
#include <iostream>
#include <vector>
#include <memory>

#include "tensor/dtensor.h"
#include "ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"

// =============================================================================
// Test: rotate3D Shard Dimension Tracking
//
// Verifies shard dimension is correctly updated after rotate3D:
//   - dim=0: transpose(1,2) → shard_dim 1↔2
//   - dim=1: transpose(0,2) → shard_dim 0↔2  
//   - dim=2: transpose(0,1) → shard_dim 0↔1
// =============================================================================

struct TestCase {
    int rotate_dim;    // Rotation axis
    int init_shard;    // Initial shard dimension
    int expect_shard;  // Expected shard dimension after rotation
    const char* name;
};

bool run_test(std::shared_ptr<DeviceMesh> mesh,
              std::shared_ptr<ProcessGroupNCCL> pg,
              const TestCase& tc) {
    std::vector<int> shape = {4, 4, 4};
    Layout layout(mesh, shape, ShardingType::SHARDED, tc.init_shard);
    
    DTensor tensor = DTensor::ones(shape, mesh, pg, layout);
    tensor.rotate3D(tc.rotate_dim, true);
    
    return tensor.get_layout().get_shard_dim() == tc.expect_shard;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    cudaSetDevice(rank);
    auto pg = init_process_group(world_size, rank);
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    
    // Define all test cases
    std::vector<TestCase> tests = {
        {0, 1, 2, "dim=0, shard 1→2"},
        {0, 2, 1, "dim=0, shard 2→1"},
        {1, 0, 2, "dim=1, shard 0→2"},
        {1, 2, 0, "dim=1, shard 2→0"},
        {2, 0, 1, "dim=2, shard 0→1"},
        {2, 1, 0, "dim=2, shard 1→0"},
        {0, 0, 0, "dim=0, shard 0 (unaffected)"},
    };
    
    int passed = 0, failed = 0;
    
    if (rank == 0) {
        std::cout << "\n=== rotate3D Shard Dimension Test ===\n\n";
    }
    
    for (const auto& tc : tests) {
        MPI_Barrier(MPI_COMM_WORLD);
        bool result = run_test(mesh, pg, tc);
        
        if (result) {
            passed++;
            if (rank == 0) std::cout << "[PASS] " << tc.name << "\n";
        } else {
            failed++;
            if (rank == 0) std::cout << "[FAIL] " << tc.name << "\n";
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\nResults: " << passed << "/" << tests.size() << " passed\n\n";
    }
    
    MPI_Finalize();
    return failed > 0 ? 1 : 0;
}

// make test_rotate3d_sharding
// mpirun -np 2 --allow-run-as-root ./test_rotate3d_sharding