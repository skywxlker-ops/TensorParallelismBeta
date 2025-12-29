#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>

#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

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

    ncclUniqueId nccl_id;
    if (rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }
    MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);

    CUDA_CHECK(cudaSetDevice(rank));

    if (rank == 0) {
        std::cout << "\nPipelined Initialization Test\n";
    }

    // Prepare data for 3 tensors
    std::vector<std::vector<float>> tensors_data;
    std::vector<Layout> layouts;
    
    if (rank == 0) {
        // Tensor 1: 4x4 row-sharded
        std::vector<float> t1(16);
        for (int i = 0; i < 16; ++i) t1[i] = static_cast<float>(i + 1);
        tensors_data.push_back(t1);
        
        // Tensor 2: 8x4 row-sharded
        std::vector<float> t2(32);
        for (int i = 0; i < 32; ++i) t2[i] = static_cast<float>(i + 100);
        tensors_data.push_back(t2);
        
        // Tensor 3: 4x4 replicated
        std::vector<float> t3(16);
        for (int i = 0; i < 16; ++i) t3[i] = static_cast<float>(i + 1000);
        tensors_data.push_back(t3);
    } else {
        // Non-root: empty vectors
        tensors_data.push_back({});
        tensors_data.push_back({});
        tensors_data.push_back({});
    }
    
    // Create layouts
    layouts.push_back(Layout(mesh, {4, 4}, ShardingType::SHARDED, 0));  // Row-sharded
    layouts.push_back(Layout(mesh, {8, 4}, ShardingType::SHARDED, 0));  // Row-sharded
    layouts.push_back(Layout(mesh, {4, 4}, ShardingType::REPLICATED));   // Replicated
    
    // Test pipelined loading
    std::vector<DTensor> tensors;
    
    auto start = Clock::now();
    DTensor::setDataFromRootPipelined(tensors_data, layouts, tensors, mesh, pg, 0);
    auto end = Clock::now();
    
    double time_ms = Duration(end - start).count();
    
    if (rank == 0) {
        std::cout << "\nLoaded 3 tensors in " << std::fixed << std::setprecision(2) 
                  << time_ms << "ms (pipelined)\n";
    }
    
    // Verify results
    bool all_pass = true;
    
    // Tensor 1: 4x4 row-sharded
    auto data1 = tensors[0].getData();
    if (rank == 0) {
        bool pass = (data1.size() == 8 && data1[0] == 1.0f && data1[7] == 8.0f);
        std::cout << "  Tensor 1 (4x4 row-sharded): " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;
    } else {
        bool pass = (data1.size() == 8 && data1[0] == 9.0f && data1[7] == 16.0f);
        std::cout << "  Tensor 1 (rank 1): " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;
    }
    
    // Tensor 2: 8x4 row-sharded
    auto data2 = tensors[1].getData();
    if (rank == 0) {
        bool pass = (data2.size() == 16 && data2[0] == 100.0f && data2[15] == 115.0f);
        std::cout << "  Tensor 2 (8x4 row-sharded): " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;
    } else {
        bool pass = (data2.size() == 16 && data2[0] == 116.0f && data2[15] == 131.0f);
        std::cout << "  Tensor 2 (rank 1): " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;
    }
    
    // Tensor 3: 4x4 replicated
    auto data3 = tensors[2].getData();
    bool pass3 = (data3.size() == 16 && data3[0] == 1000.0f && data3[15] == 1015.0f);
    if (rank == 0) {
        std::cout << "  Tensor 3 (4x4 replicated): " << (pass3 ? "PASS" : "FAIL") << std::endl;
    }
    all_pass &= pass3;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n" << (all_pass ? "All tests passed!" : "Some tests failed!") << "\n";
        std::cout << "\nDone.\n";
    }

    MPI_Finalize();
    return all_pass ? 0 : 1;
}
