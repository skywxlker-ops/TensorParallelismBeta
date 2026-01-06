/**
 * DTensor Creation Methods Demo
 * 
 * Demonstrates the 3 PyTorch-style creation methods:
 * 1. from_local()       - Create from existing local tensor shard
 * 2. distribute_tensor()- Distribute an existing GPU tensor
 * 3. setDataFromRoot()  - Load from host data (legacy)
 * 
 * Run: mpirun -np 2 --allow-run-as-root ./test_dtensor_creation_methods
 */

#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <nccl.h>
#include <iomanip>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    cudaSetDevice(rank);
    
    // Setup
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    
    if (rank == 0) {
        std::cout << "\n=== DTensor Creation Methods (" << world_size << " GPUs) ===" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // METHOD 1: from_local() - Create from existing local tensor
    // =========================================================================
    {
        std::vector<int> global_shape = {4, 8};
        Layout layout(mesh, global_shape, ShardingType::SHARDED, 0);
        std::vector<int> local_shape = layout.get_local_shape(rank);
        
        OwnTensor::Shape shape_obj;
        shape_obj.dims.assign(local_shape.begin(), local_shape.end());
        OwnTensor::TensorOptions opts = OwnTensor::TensorOptions()
            .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
            .with_dtype(OwnTensor::Dtype::Float32);
        
        OwnTensor::Tensor local_tensor = OwnTensor::Tensor::full(shape_obj, opts, (rank + 1) * 10.0f);
        auto dt = DTensor::from_local(local_tensor, mesh, pg, layout);
        
        std::cout << "[Rank " << rank << "] from_local()        -> value: " << dt.getData()[0] << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // METHOD 2: distribute_tensor() - Distribute existing GPU tensor
    // =========================================================================
    {
        std::vector<int> global_shape = {4, 8};
        
        OwnTensor::Shape shape_obj;
        shape_obj.dims.assign(global_shape.begin(), global_shape.end());
        OwnTensor::TensorOptions opts = OwnTensor::TensorOptions()
            .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
            .with_dtype(OwnTensor::Dtype::Float32);
        OwnTensor::Tensor global_tensor = OwnTensor::Tensor::full(shape_obj, opts, 99.0f);
        
        // Sharded
        Layout sharded_layout(mesh, global_shape, ShardingType::SHARDED, 0);
        auto dt_sharded = DTensor::distribute_tensor(global_tensor, mesh, pg, sharded_layout, 0);
        std::cout << "[Rank " << rank << "] distribute(SHARDED) -> size: " << dt_sharded.getData().size() << std::endl;
        
        // Replicated
        Layout rep_layout = Layout::replicated(mesh, global_shape);
        auto dt_rep = DTensor::distribute_tensor(global_tensor, mesh, pg, rep_layout, 0);
        std::cout << "[Rank " << rank << "] distribute(REPLICATED) -> size: " << dt_rep.getData().size() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // METHOD 3: setDataFromRoot() - Load from host vector
    // =========================================================================
    {
        std::vector<int> global_shape = {4, 8};
        Layout layout(mesh, global_shape, ShardingType::SHARDED, 0);
        
        std::vector<float> host_data;
        if (rank == 0) {
            host_data.resize(32);
            for (int i = 0; i < 32; i++) host_data[i] = (float)i;
        }
        
        DTensor dt(mesh, pg);
        dt.setDataFromRoot(host_data, layout, 0);
        std::cout << "[Rank " << rank << "] setDataFromRoot()   -> first: " << dt.getData()[0] << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n=== All methods work! ===" << std::endl;
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
 *   make lib                           # Build library (if needed)
 *   make test_dtensor_creation_methods # Build this test
 *   mpirun -np 2 ./tests/test_dtensor_creation_methods
 * 
 * ============================================================================
 */
