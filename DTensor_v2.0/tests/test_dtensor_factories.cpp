/**
 * Test DTensor Factory Functions
 * 
 * Tests: empty, zeros, ones, full, rand, randn, randint, from_local
 * Run: mpirun -np 2 --allow-run-as-root ./tests/test_dtensor_factories
 */

#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <nccl.h>
#include <cmath>

void test_empty(std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    std::vector<int> global_shape = {4, 8};
    Layout layout(mesh, global_shape, ShardingType::SHARDED, 0);  // Shard on dim 0
    
    auto dt = DTensor::empty(global_shape, mesh, pg, layout);
    
    std::vector<int> expected_local_shape = {2, 8};  // 4/2 = 2 rows per rank
    auto local_shape = layout.get_local_shape(rank);
    
    bool pass = (local_shape == expected_local_shape);
    std::cout << "[Rank " << rank << "] DTensor::empty - local shape [" 
              << local_shape[0] << ", " << local_shape[1] << "] "
              << (pass ? "PASS" : "FAIL") << std::endl;
}

void test_zeros(std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    std::vector<int> global_shape = {4, 6};
    Layout layout(mesh, global_shape, ShardingType::SHARDED, 1);  // Shard on dim 1
    
    auto dt = DTensor::zeros(global_shape, mesh, pg, layout);
    
    auto data = dt.getData();
    bool all_zeros = true;
    for (float v : data) {
        if (v != 0.0f) { all_zeros = false; break; }
    }
    
    std::cout << "[Rank " << rank << "] DTensor::zeros - all zeros: " 
              << (all_zeros ? "PASS" : "FAIL") << std::endl;
}

void test_ones(std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    std::vector<int> global_shape = {4, 4};
    Layout layout = Layout::replicated(mesh, global_shape);  // Replicated
    
    auto dt = DTensor::ones(global_shape, mesh, pg, layout);
    
    auto data = dt.getData();
    bool all_ones = true;
    for (float v : data) {
        if (v != 1.0f) { all_ones = false; break; }
    }
    
    std::cout << "[Rank " << rank << "] DTensor::ones - all ones: " 
              << (all_ones ? "PASS" : "FAIL") << std::endl;
}

void test_full(std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    std::vector<int> global_shape = {2, 4};
    Layout layout(mesh, global_shape, ShardingType::SHARDED, 0);
    float fill_value = 3.14f;
    
    auto dt = DTensor::full(global_shape, fill_value, mesh, pg, layout);
    
    auto data = dt.getData();
    bool all_correct = true;
    for (float v : data) {
        if (std::abs(v - fill_value) > 1e-5f) { all_correct = false; break; }
    }
    
    std::cout << "[Rank " << rank << "] DTensor::full(3.14) - correct: " 
              << (all_correct ? "PASS" : "FAIL") << std::endl;
}

void test_rand(std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    std::vector<int> global_shape = {100, 100};
    Layout layout(mesh, global_shape, ShardingType::SHARDED, 0);
    
    auto dt = DTensor::rand(global_shape, mesh, pg, layout);
    
    auto data = dt.getData();
    bool in_range = true;
    float sum = 0;
    for (float v : data) {
        if (v < 0.0f || v >= 1.0f) { in_range = false; break; }
        sum += v;
    }
    float mean = sum / data.size();
    // Mean should be ~0.5 for uniform [0, 1)
    bool mean_ok = (mean > 0.4f && mean < 0.6f);
    
    std::cout << "[Rank " << rank << "] DTensor::rand - in [0,1): " 
              << (in_range ? "PASS" : "FAIL")
              << ", mean=" << mean << " " << (mean_ok ? "PASS" : "FAIL") << std::endl;
}

void test_randn(std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    std::vector<int> global_shape = {100, 100};
    Layout layout(mesh, global_shape, ShardingType::SHARDED, 0);
    
    auto dt = DTensor::randn(global_shape, mesh, pg, layout);
    
    auto data = dt.getData();
    float sum = 0, sum_sq = 0;
    for (float v : data) {
        sum += v;
        sum_sq += v * v;
    }
    float mean = sum / data.size();
    float variance = (sum_sq / data.size()) - (mean * mean);
    
    // Mean should be ~0, variance ~1 for N(0,1)
    bool mean_ok = (std::abs(mean) < 0.2f);
    bool var_ok = (variance > 0.7f && variance < 1.3f);
    
    std::cout << "[Rank " << rank << "] DTensor::randn - mean=" << mean 
              << " " << (mean_ok ? "PASS" : "FAIL")
              << ", var=" << variance << " " << (var_ok ? "PASS" : "FAIL") << std::endl;
}

void test_randint(std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    std::vector<int> global_shape = {4, 4};
    Layout layout(mesh, global_shape, ShardingType::SHARDED, 0);
    int64_t low = 0, high = 10;
    
    auto dt = DTensor::randint(low, high, global_shape, mesh, pg, layout);
    
    auto data = dt.getData();
    bool in_range = true;
    for (float v : data) {
        int iv = static_cast<int>(v);
        if (iv < low || iv >= high) { in_range = false; break; }
    }
    
    std::cout << "[Rank " << rank << "] DTensor::randint [0,10) - in range: " 
              << (in_range ? "PASS" : "FAIL") << std::endl;
}

void test_from_local(std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    std::vector<int> global_shape = {4, 4};
    Layout layout(mesh, global_shape, ShardingType::SHARDED, 0);
    
    // Create local tensor manually
    std::vector<int> local_shape = layout.get_local_shape(rank);
    OwnTensor::Shape shape_obj;
    shape_obj.dims.assign(local_shape.begin(), local_shape.end());
    OwnTensor::TensorOptions opts;
    opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
               .with_dtype(OwnTensor::Dtype::Float32);
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::full(shape_obj, opts, rank + 1.0f);
    
    auto dt = DTensor::from_local(local_tensor, mesh, pg, layout);
    
    auto data = dt.getData();
    bool correct = true;
    for (float v : data) {
        if (std::abs(v - (rank + 1.0f)) > 1e-5f) { correct = false; break; }
    }
    
    std::cout << "[Rank " << rank << "] DTensor::from_local - value=" << (rank + 1) 
              << ": " << (correct ? "PASS" : "FAIL") << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    cudaSetDevice(rank);
    
    // Setup device mesh following test_mlp_benchmark.cpp pattern
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    
    
    auto pg = init_process_group(world_size, rank);
    
    std::cout << "\n=== DTensor Factory Functions Test (Rank " << rank << "/" << world_size << ") ===\n" << std::endl;
    
    test_empty(mesh, pg);
    MPI_Barrier(MPI_COMM_WORLD);
    
    test_zeros(mesh, pg);
    MPI_Barrier(MPI_COMM_WORLD);
    
    test_ones(mesh, pg);
    MPI_Barrier(MPI_COMM_WORLD);
    
    test_full(mesh, pg);
    MPI_Barrier(MPI_COMM_WORLD);
    
    test_rand(mesh, pg);
    MPI_Barrier(MPI_COMM_WORLD);
    
    test_randn(mesh, pg);
    MPI_Barrier(MPI_COMM_WORLD);
    
    test_randint(mesh, pg);
    MPI_Barrier(MPI_COMM_WORLD);
    
    test_from_local(mesh, pg);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test distribute_tensor (new)
    {
        int rank = pg->get_rank();
        std::vector<int> global_shape = {4, 8};
        
        // Create global tensor on root
        OwnTensor::Shape shape_obj;
        shape_obj.dims.assign(global_shape.begin(), global_shape.end());
        OwnTensor::TensorOptions opts;
        opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
                   .with_dtype(OwnTensor::Dtype::Float32);
        OwnTensor::Tensor global_tensor = OwnTensor::Tensor::full(shape_obj, opts, 42.0f);
        
        // Test 1: Distribute as sharded (dim 0)
        Layout sharded_layout(mesh, global_shape, ShardingType::SHARDED, 0);
        auto dt_sharded = DTensor::distribute_tensor(global_tensor, mesh, pg, sharded_layout, 0);
        
        auto sharded_data = dt_sharded.getData();
        bool sharded_ok = true;
        for (float v : sharded_data) {
            if (std::abs(v - 42.0f) > 1e-5f) { sharded_ok = false; break; }
        }
        
        // Test 2: Distribute as replicated
        Layout rep_layout = Layout::replicated(mesh, global_shape);
        auto dt_rep = DTensor::distribute_tensor(global_tensor, mesh, pg, rep_layout, 0);
        
        auto rep_data = dt_rep.getData();
        bool rep_ok = (rep_data.size() == 4 * 8);
        for (float v : rep_data) {
            if (std::abs(v - 42.0f) > 1e-5f) { rep_ok = false; break; }
        }
        
        std::cout << "[Rank " << rank << "] DTensor::distribute_tensor - sharded: "
                  << (sharded_ok ? "PASS" : "FAIL")
                  << ", replicated: " << (rep_ok ? "PASS" : "FAIL") << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n=== All Factory Tests Complete ===\n" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
