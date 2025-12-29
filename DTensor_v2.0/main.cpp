#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include "bridge/tensor_ops_bridge.h"
#include "memory/cachingAllocator.hpp"

// Helper function to compare vectors
bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b, int rank) {
    if (a.size() != b.size()) {
        std::cerr << "[Rank " << rank << "] Vector sizes don't match! "
                  << a.size() << " vs " << b.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 1e-5) {
            std::cerr << "[Rank " << rank << "] Mismatch at index " << i << ": "
                      << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv) {
    // MPI_Init(&argc, &argv);

    // int rank, world_size;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // if (world_size != 2) {
    //     if (rank == 0) {
    //         std::cerr << "This test requires exactly 2 MPI processes." << std::endl;
    //     }
    //     MPI_Finalize();
    //     return 1;
    // }

    // // 1. Setup
    // ncclUniqueId nccl_id;
    // if (rank == 0) {
    //     ncclGetUniqueId(&nccl_id);
    // }
    // MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    // auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});

    // // Global tensor shape and data
    // std::vector<int> global_shape = {4, 8};
    // size_t global_numel = global_shape[0] * global_shape[1];
    // std::vector<float> global_data(global_numel);
    // std::iota(global_data.begin(), global_data.end(), 0);

    // if (rank == 0) {
    //     std::cout << "--- C++ Test Setup Complete ---" << std::endl;
    // }

    // // =================================================================
    // // Test 1: Sharded to Replicated
    // // =================================================================
    // if (rank == 0) {
    //     std::cout << "\n--- Running Test 1: Sharded to Replicated ---" << std::endl;
    // }

    // Layout layout_sharded_col({mesh, global_shape, ShardingType::SHARDED, 1});
    
    // // Manually calculate the local shard for each rank (column sharding)
    // std::vector<float> local_data_shard_col;
    // for (int r = 0; r < global_shape[0]; ++r) {
    //     for (int c = rank * (global_shape[1]/world_size); c < (rank + 1) * (global_shape[1]/world_size); ++c) {
    //         local_data_shard_col.push_back(global_data[r * global_shape[1] + c]);
    //     }
    // }
    
    // DTensor tensor_sharded(mesh, pg);
    // tensor_sharded.setData(local_data_shard_col, layout_sharded_col);

    // Layout layout_replicated({mesh, global_shape, ShardingType::REPLICATED});
    // DTensor tensor_replicated = tensor_sharded.redistribute(layout_replicated);

    // // Verification
    // assert(tensor_replicated.get_layout().is_replicated());
    // std::vector<float> replicated_data = tensor_replicated.getData();
    // assert(compare_vectors(replicated_data, global_data, rank));
    
    // if (rank == 0) {
    //     std::cout << "PASSED: Sharded -> Replicated" << std::endl;
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    // // =================================================================
    // // Test 2: Replicated to Sharded
    // // =================================================================
    // if (rank == 0) {
    //     std::cout << "\n--- Running Test 2: Replicated to Sharded ---" << std::endl;
    // }

    // DTensor tensor_resharded = tensor_replicated.redistribute(layout_sharded_col);

    // // Verification
    // assert(tensor_resharded.get_layout().is_sharded());
    // std::vector<float> resharded_data = tensor_resharded.getData();
    // assert(compare_vectors(resharded_data, local_data_shard_col, rank));

    // if (rank == 0) {
    //     std::cout << "PASSED: Replicated -> Sharded" << std::endl;
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    // // =================================================================
    // // Test 3: Sharded to Sharded (different dimension)
    // // =================================================================
    // if (rank == 0) {
    //     std::cout << "\n--- Running Test 3: Sharded to Sharded (different dim) ---" << std::endl;
    // }
    
    // Layout layout_sharded_row({mesh, global_shape, ShardingType::SHARDED, 0});
    
    // // Manually calculate the new expected local shard (row sharding)
    // std::vector<float> local_data_shard_row;
    //  for (int r = rank * (global_shape[0]/world_size); r < (rank + 1) * (global_shape[0]/world_size); ++r) {
    //     for (int c = 0; c < global_shape[1]; ++c) {
    //         local_data_shard_row.push_back(global_data[r * global_shape[1] + c]);
    //     }
    // }

    // DTensor tensor_row_sharded = tensor_sharded.redistribute(layout_sharded_row);

    // // Verification
    // assert(tensor_row_sharded.get_layout().is_sharded());
    // std::vector<float> row_sharded_data = tensor_row_sharded.getData();
    // assert(compare_vectors(row_sharded_data, local_data_shard_row, rank));

    // if (rank == 0) {
    //     std::cout << "PASSED: Sharded -> Sharded (col to row)" << std::endl;
    //     std::cout << "\n--- All C++ tests completed successfully ---" << std::endl;
    // }

    // MPI_Finalize();
    return 0;
}
