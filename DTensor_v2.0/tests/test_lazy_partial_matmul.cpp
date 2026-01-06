#include <iostream>
#include <vector>
#include <cmath>
#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"

/**
 * Test Lazy Partial in Row-Parallel Matmul:
 * - X [2, 4] sharded on dim 1 @ W [4, 2] sharded on dim 0 -> Y_partial [2, 2]
 * - Y_partial should be Partial layout (not immediately reduced)
 * - redistribute() to Replicated triggers AllReduce
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    cudaSetDevice(rank);
    auto pg = init_process_group(world_size, rank);
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});

    // X: [2, 4] sharded on dim 1 -> each GPU has [2, 2]
    std::vector<int> X_shape = {2, 4};
    std::vector<float> X_local = {1, 2, 3, 4};  // Each GPU has same local data for simplicity
    Layout X_layout(mesh, X_shape, ShardingType::SHARDED, 1);
    
    DTensor X(mesh, pg);
    X.setData(X_local, X_layout);

    // W: [4, 2] sharded on dim 0 -> each GPU has [2, 2]
    std::vector<int> W_shape = {4, 2};
    std::vector<float> W_local = {1, 0, 0, 1};  // Identity-like for easy verification
    Layout W_layout(mesh, W_shape, ShardingType::SHARDED, 0);
    
    DTensor W(mesh, pg);
    W.setData(W_local, W_layout);

    // Row-parallel matmul -> should return PARTIAL (lazy)
    DTensor Y_partial = X.matmul(W);

    // Check: Y should be Partial layout (not reduced yet!)
    bool is_partial = Y_partial.get_layout().is_partial();

    if (rank == 0) {
        std::cout << "Row-parallel matmul result:" << std::endl;
        std::cout << "  Layout is Partial: " << (is_partial ? "YES" : "NO") << std::endl;
    }

    // Now reduce to Replicated
    Layout rep_layout(mesh, {2, 2}, ShardingType::REPLICATED);
    DTensor Y_reduced = Y_partial.redistribute(rep_layout);
    
    std::vector<float> result = Y_reduced.getData();

    if (rank == 0) {
        std::cout << "  After redistribute (AllReduce):" << std::endl;
        std::cout << "    Y = [[" << result[0] << ", " << result[1] << "], ["
                  << result[2] << ", " << result[3] << "]]" << std::endl;
        std::cout << (is_partial ? "PASSED: Lazy Partial works!" : "FAILED: Not Partial") << std::endl;
    }

    MPI_Finalize();
    return is_partial ? 0 : 1;
}

/*
 * ============================================================================
 * BUILD & RUN INSTRUCTIONS
 * ============================================================================
 * 
 * This test is not currently in the Makefile. To add it:
 * 
 *   # Add to Makefile following the pattern for other tests, or
 *   # compile manually:
 *   
 *   make lib
 *   mpic++ -std=c++17 -O3 -fPIC -g -DWITH_CUDA \
 *       -I. -I./tensor -I./process_group -I./memory -I./bridge \
 *       -I./Tensor-Implementations/include -I/usr/local/cuda/include \
 *       tests/test_lazy_partial_matmul.cpp -o tests/test_lazy_partial_matmul \
 *       lib/unparalleled.a Tensor-Implementations/lib/libtensor.a \
 *       -L/usr/local/cuda/lib64 -lnccl -lcudart -lcublas -lcurand
 *   
 *   mpirun -np 2 ./tests/test_lazy_partial_matmul
 * 
 * ============================================================================
 */
