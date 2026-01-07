/**
 * Test: Verify public headers work correctly
 * 
 * This test includes only the umbrella header to verify the public API.
 */

#include <unparalleled/unparalleled.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    cudaSetDevice(rank);
    
    // Test DeviceMesh
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    
    // Test ProcessGroupNCCL
    auto pg = init_process_group(world_size, rank);
    
    // Test Layout
    std::vector<int> shape = {4, 8};
    Layout layout(mesh, shape, ShardingType::SHARDED, 0);
    
    // Test DTensor factory
    auto tensor = DTensor::zeros(shape, mesh, pg, layout);
    
    if (rank == 0) {
        std::cout << "Public headers test PASSED" << std::endl;
        std::cout << "  - DeviceMesh: created" << std::endl;
        std::cout << "  - ProcessGroupNCCL: initialized" << std::endl;
        std::cout << "  - Layout: SHARDED on dim 0" << std::endl;
        std::cout << "  - DTensor::zeros: created" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}

/*
 * ============================================================================
 * BUILD & RUN INSTRUCTIONS
 * ============================================================================
 * 
 * From DTensor_v2.0 directory:
 * 
 *   make lib
 *   make test_public_headers
 *   mpirun -np 2 ./tests/test_public_headers
 * 
 * ============================================================================
 */

