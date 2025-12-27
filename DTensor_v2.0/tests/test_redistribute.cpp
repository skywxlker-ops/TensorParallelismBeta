#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"

// Helper function to compare vectors
bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b, int rank, float tol = 1e-5) {
    if (a.size() != b.size()) {
        std::cerr << "[Rank " << rank << "] Vector sizes don't match! "
                  << a.size() << " vs " << b.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) {
            std::cerr << "[Rank " << rank << "] Mismatch at index " << i << ": "
                      << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "This test requires exactly 2 MPI processes." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // 1. Setup with ProcessGroupNCCL
    cudaSetDevice(rank);  // Set correct GPU
    auto pg = init_process_group(world_size, rank);
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});

    // Global tensor shape and data
    std::vector<int> global_shape = {4, 8};
    size_t global_numel = global_shape[0] * global_shape[1];
    std::vector<float> global_data(global_numel);
    std::iota(global_data.begin(), global_data.end(), 0);

    // if (rank == 0) {
    //     std::cout << "--- C++ Redistribute Test Suite ---" << std::endl;
    // }

    // =================================================================
    // Test 1: Sharded to Replicated
    // =================================================================
    if (rank == 0) {
        std::cout << "\n--- Test 1: Sharded -> Replicated ---" << std::endl;
    }

    Layout layout_sharded_col(mesh, global_shape, ShardingType::SHARDED, 1);
    
    // Manually calculate the local shard for each rank (column sharding)
    std::vector<float> local_data_shard_col;
    for (int r = 0; r < global_shape[0]; ++r) {
        for (int c = rank * (global_shape[1]/world_size); c < (rank + 1) * (global_shape[1]/world_size); ++c) {
            local_data_shard_col.push_back(global_data[r * global_shape[1] + c]);
        }
    }
    
    DTensor tensor_sharded(mesh, pg);
    tensor_sharded.setData(local_data_shard_col, layout_sharded_col);

    Layout layout_replicated(mesh, global_shape, ShardingType::REPLICATED);
    DTensor tensor_replicated = tensor_sharded.redistribute(layout_replicated);

    // Verification
    assert(tensor_replicated.get_layout().is_replicated());
    std::vector<float> replicated_data = tensor_replicated.getData();
    assert(compare_vectors(replicated_data, global_data, rank));
    
    if (rank == 0) {
        std::cout << "PASSED: Sharded -> Replicated" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =================================================================
    // Test 2: Replicated to Sharded
    // =================================================================
    if (rank == 0) {
        std::cout << "\n--- Test 2: Replicated -> Sharded ---" << std::endl;
    }

    DTensor tensor_resharded = tensor_replicated.redistribute(layout_sharded_col);

    // Verification
    assert(tensor_resharded.get_layout().is_sharded());
    std::vector<float> resharded_data = tensor_resharded.getData();
    assert(compare_vectors(resharded_data, local_data_shard_col, rank));

    if (rank == 0) {
        std::cout << "PASSED: Replicated -> Sharded" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =================================================================
    // Test 3: Sharded to Sharded (different dimension)
    // =================================================================
    if (rank == 0) {
        std::cout << "\n--- Test 3: Sharded -> Sharded (different dim) ---" << std::endl;
    }
    
    Layout layout_sharded_row(mesh, global_shape, ShardingType::SHARDED, 0);
    
    // Manually calculate the new expected local shard (row sharding)
    std::vector<float> local_data_shard_row;
     for (int r = rank * (global_shape[0]/world_size); r < (rank + 1) * (global_shape[0]/world_size); ++r) {
        for (int c = 0; c < global_shape[1]; ++c) {
            local_data_shard_row.push_back(global_data[r * global_shape[1] + c]);
        }
    }

    DTensor tensor_row_sharded = tensor_sharded.redistribute(layout_sharded_row);

    // Verification
    assert(tensor_row_sharded.get_layout().is_sharded());
    std::vector<float> row_sharded_data = tensor_row_sharded.getData();
    assert(compare_vectors(row_sharded_data, local_data_shard_row, rank));

    if (rank == 0) {
        std::cout << "PASSED: Sharded -> Sharded (col to row)" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =================================================================
    // Test 4: Partial to Replicated (AllReduce)
    // =================================================================
    if (rank == 0) {
        std::cout << "\n--- Test 4: Partial -> Replicated (AllReduce) ---" << std::endl;
    }
    
    // Create a PARTIAL layout - each rank has a portion of the sum
    Layout layout_partial(mesh, global_shape, ShardingType::PARTIAL, -1, "sum");
    
    // Each rank gets partial data: final sum should equal global_data * world_size
    // So we give each rank the full global_data, and after AllReduce we expect global_data * 2
    DTensor tensor_partial(mesh, pg);
    tensor_partial.setData(global_data, layout_partial);
    
    // Redistribute Partial -> Replicated (triggers AllReduce)
    DTensor tensor_from_partial = tensor_partial.redistribute(layout_replicated);
    
    // Expected: sum of all partial values = global_data * world_size
    std::vector<float> expected_sum(global_numel);
    for (size_t i = 0; i < global_numel; ++i) {
        expected_sum[i] = global_data[i] * world_size;
    }
    
    // Verification
    assert(tensor_from_partial.get_layout().is_replicated());
    std::vector<float> reduced_data = tensor_from_partial.getData();
    assert(compare_vectors(reduced_data, expected_sum, rank));
    
    if (rank == 0) {
        std::cout << "PASSED: Partial -> Replicated (AllReduce)" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =================================================================
    // Test 5: Replicated to Partial (Partition)
    // =================================================================
    if (rank == 0) {
        std::cout << "\n--- Test 5: Replicated -> Partial (Partition) ---" << std::endl;
    }
    
    // Create replicated tensor with global_data
    DTensor tensor_rep(mesh, pg);
    tensor_rep.setData(global_data, layout_replicated);
    
    // Redistribute Replicated -> Partial (should divide by world_size for "sum")
    DTensor tensor_partitioned = tensor_rep.redistribute(layout_partial);
    
    // Expected: partitioned_value = global_data / world_size
    std::vector<float> expected_partitioned(global_numel);
    for (size_t i = 0; i < global_numel; ++i) {
        expected_partitioned[i] = global_data[i] / world_size;
    }
    
    // Verification
    assert(tensor_partitioned.get_layout().is_partial());
    std::vector<float> partitioned_data = tensor_partitioned.getData();
    assert(compare_vectors(partitioned_data, expected_partitioned, rank));
    
    if (rank == 0) {
        std::cout << "PASSED: Replicated -> Partial (Partition)" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =================================================================
    // Test 6: Partial to Shard (ReduceScatter)
    // =================================================================
    if (rank == 0) {
        std::cout << "\n--- Test 6: Partial -> Shard (ReduceScatter) ---" << std::endl;
    }
    
    // Create partial tensor - each rank has global_data
    DTensor tensor_partial2(mesh, pg);
    tensor_partial2.setData(global_data, layout_partial);
    
    // Redistribute Partial -> Sharded on dim 0 (triggers ReduceScatter)
    DTensor tensor_reduced_sharded = tensor_partial2.redistribute(layout_sharded_row);
    
    // Expected: row shard of (global_data * world_size)
    std::vector<float> expected_reduced_shard;
    for (int r = rank * (global_shape[0]/world_size); r < (rank + 1) * (global_shape[0]/world_size); ++r) {
        for (int c = 0; c < global_shape[1]; ++c) {
            expected_reduced_shard.push_back(global_data[r * global_shape[1] + c] * world_size);
        }
    }
    
    // Verification
    assert(tensor_reduced_sharded.get_layout().is_sharded());
    std::vector<float> reduced_sharded_data = tensor_reduced_sharded.getData();
    assert(compare_vectors(reduced_sharded_data, expected_reduced_shard, rank));
    
    if (rank == 0) {
        std::cout << "PASSED: Partial -> Shard (ReduceScatter)" << std::endl;
        std::cout << "\n--- All Redistribute Tests PASSED! ---" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
