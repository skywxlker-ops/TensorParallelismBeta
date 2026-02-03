/**
 * Example: Using sync() for gradient synchronization in distributed training
 * 
 * This demonstrates how to use the DTensor::sync() function to average
 * gradients across all GPUs during backpropagation.
 */

#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <nccl.h>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Create device mesh (1D mesh with all GPUs)
    auto device_mesh = std::make_shared<DeviceMesh>(
        std::vector<int>{world_size}
    );
    
    // Create NCCL communicator
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    
    // Create DTensor for gradients (replicated across all GPUs)
    DTensor gradients(device_mesh, pg);
    
    // Simulate local gradient computation on each GPU
    // Each GPU computes different gradients based on its local data
    std::vector<float> local_grads(1024);
    for (int i = 0; i < 1024; i++) {
        local_grads[i] = static_cast<float>(rank + 1) * 0.1f;  // Different per GPU
    }
    
    // Set local gradients
    Layout replicated_layout = Layout::replicated(device_mesh, {1024});
    gradients.setData(local_grads, replicated_layout);
    
    if (rank == 0) {
        std::cout << "Before sync() - GPU 0 gradients (first 5): ";
        auto data = gradients.getData();
        for (int i = 0; i < 5; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Synchronize gradients across all GPUs
    // This performs AllReduce with ncclAvg: gradients = avg(all_grads)
    gradients.sync();
    
    if (rank == 0) {
        std::cout << "After sync() - GPU 0 gradients (first 5): ";
        auto data = gradients.getData();
        for (int i = 0; i < 5; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Gradients averaged across all " << world_size << " GPUs" << std::endl;
    }
    
    // Use averaged gradients for weight update
    // optimizer.step(gradients);
    
    MPI_Finalize();
    return 0;
}

/**
 * Expected behavior with 4 GPUs:
 * 
 * Before sync():
 * - GPU 0: all values = 0.1
 * - GPU 1: all values = 0.2  
 * - GPU 2: all values = 0.3
 * - GPU 3: all values = 0.4
 * 
 * After sync():
 * - All GPUs: all values = (0.1 + 0.2 + 0.3 + 0.4) / 4 = 0.25
 */
