#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <cmath>

// === DTensor Core ===
#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"

// === TensorLib (for ops) ===
#include "bridge/tensor_ops_bridge.h"
#include "memory/cachingAllocator.hpp"

using namespace OwnTensor;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// ============================================================================
// Test: GPU-Native Initialization
// ============================================================================

void print_test(int rank, const std::string& name) {
    if (rank == 0) {
        std::cout << "\n[" << name << "]" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

size_t get_gpu_memory_used() {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return total - free;
}

bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b, int rank, float tolerance = 1e-5) {
    if (a.size() != b.size()) {
        std::cerr << "[Rank " << rank << "] Size mismatch: " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cerr << "[Rank " << rank << "] Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

void test_replicated_init(int rank, int world_size, 
                           std::shared_ptr<DeviceMesh> mesh, 
                           std::shared_ptr<ProcessGroup> pg) {
    print_test(rank, "Replicated");
    
    std::vector<int> global_shape = {4, 4};
    Layout layout_replicated(mesh, global_shape, ShardingType::REPLICATED);
    
    std::vector<float> full_data;
    if (rank == 0) {
        full_data.resize(16);
        for (int i = 0; i < 16; ++i) {
            full_data[i] = static_cast<float>(i + 1);
        }
    }
    
    DTensor tensor(mesh, pg);
    tensor.setDataFromRoot(full_data, layout_replicated, 0);
    
    std::vector<float> result = tensor.getData();
    std::vector<float> expected(16);
    for (int i = 0; i < 16; ++i) {
        expected[i] = static_cast<float>(i + 1);
    }
    
    if (compare_vectors(result, expected, rank)) {
        if (rank == 0) std::cout << "  PASS" << std::endl;
    } else {
        std::cerr << "  FAIL (rank " << rank << ")" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

void test_row_sharded_init(int rank, int world_size, 
                            std::shared_ptr<DeviceMesh> mesh, 
                            std::shared_ptr<ProcessGroup> pg) {
    print_test(rank, "Row-Sharded");
    
    std::vector<int> global_shape = {8, 4};
    Layout layout_row_sharded(mesh, global_shape, ShardingType::SHARDED, 0);
    
    std::vector<float> full_data;
    if (rank == 0) {
        full_data.resize(32);
        for (int i = 0; i < 32; ++i) {
            full_data[i] = static_cast<float>(i + 1);
        }
    }
    
    DTensor tensor(mesh, pg);
    tensor.setDataFromRoot(full_data, layout_row_sharded, 0);
    
    std::vector<float> result = tensor.getData();
    std::vector<float> expected;
    if (rank == 0) {
        expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    } else if (rank == 1 && world_size > 1) {
        expected = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    }
    
    if (compare_vectors(result, expected, rank)) {
        std::cout << "  rank " << rank << ": PASS" << std::endl;
    } else {
        std::cerr << "  rank " << rank << ": FAIL" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

void test_col_sharded_init(int rank, int world_size, 
                            std::shared_ptr<DeviceMesh> mesh, 
                            std::shared_ptr<ProcessGroup> pg) {
    print_test(rank, "Column-Sharded");
    
    std::vector<int> global_shape = {4, 8};
    Layout layout_col_sharded(mesh, global_shape, ShardingType::SHARDED, 1);
    
    std::vector<float> full_data;
    if (rank == 0) {
        full_data.resize(32);
        for (int i = 0; i < 32; ++i) {
            full_data[i] = static_cast<float>(i + 1);
        }
    }
    
    DTensor tensor(mesh, pg);
    tensor.setDataFromRoot(full_data, layout_col_sharded, 0);
    
    std::vector<float> result = tensor.getData();
    std::vector<float> expected;
    if (rank == 0) {
        expected = {1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 28};
    } else if (rank == 1 && world_size > 1) {
        expected = {5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32};
    }
    
    if (compare_vectors(result, expected, rank)) {
        std::cout << "  rank " << rank << ": PASS" << std::endl;
    } else {
        std::cerr << "  rank " << rank << ": FAIL" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

void test_memory_benchmark(int rank, int world_size,
                            std::shared_ptr<DeviceMesh> mesh,
                            std::shared_ptr<ProcessGroup> pg) {
    if (rank == 0) std::cout << "\n[Memory Benchmark]" << std::endl;
    
    // Use larger tensor for visible memory difference
    std::vector<int> global_shape = {1024, 1024};  // 1M floats = 4MB
    Layout layout_row_sharded(mesh, global_shape, ShardingType::SHARDED, 0);
    
    size_t before = get_gpu_memory_used();
    
    std::vector<float> full_data;
    if (rank == 0) {
        full_data.resize(1024 * 1024);
        for (size_t i = 0; i < full_data.size(); ++i) {
            full_data[i] = static_cast<float>(i % 100);
        }
    }
    
    DTensor tensor(mesh, pg);
    tensor.setDataFromRoot(full_data, layout_row_sharded, 0);
    
    size_t after = get_gpu_memory_used();
    size_t used = after - before;
    
    std::cout << "  rank " << rank << ": used " 
              << (used / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
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

    // Setup NCCL
    ncclUniqueId nccl_id;
    if (rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }
    MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Create DeviceMesh and ProcessGroup
    std::vector<int> mesh_shape = {world_size};
    auto mesh = std::make_shared<DeviceMesh>(mesh_shape);
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);

    CUDA_CHECK(cudaSetDevice(rank));

    if (rank == 0) {
        std::cout << "\nGPU-Native Init Tests:\n";
    }

    test_replicated_init(rank, world_size, mesh, pg);
    test_row_sharded_init(rank, world_size, mesh, pg);
    test_col_sharded_init(rank, world_size, mesh, pg);
    test_memory_benchmark(rank, world_size, mesh, pg);

    if (rank == 0) {
        std::cout << "\nDone.\n";
    }

    MPI_Finalize();
    return 0;
}
