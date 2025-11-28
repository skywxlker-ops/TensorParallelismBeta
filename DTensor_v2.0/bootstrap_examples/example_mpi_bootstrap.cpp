/*
 * NCCL Bootstrap Example 1: MPI Bootstrap
 * 
 * This example demonstrates using MPI for NCCL bootstrap.
 * MPI handles the out-of-band communication needed to distribute
 * the ncclUniqueId to all ranks.
 * 
 * Advantages:
 *   - Simple and robust
 *   - MPI handles process management
 *   - Works across nodes automatically
 * 
 * Disadvantages:
 *   - Requires MPI installation
 *   - Additional dependency
 */

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define NCCL_CHECK(call) do { \
    ncclResult_t res = call; \
    if (res != ncclSuccess) { \
        std::cerr << "[NCCL Error] " << ncclGetErrorString(res) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "[CUDA Error] " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

int main(int argc, char** argv) {
    // ========================================================================
    // STEP 1: Initialize MPI
    // ========================================================================
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // ========================================================================
    // STEP 2: Set CUDA device (one GPU per MPI rank)
    // ========================================================================
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    
    // Use modulo to handle cases where world_size > num_devices
    int device_id = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    std::cout << "[Rank " << rank << "] Assigned to GPU " << device_id << std::endl;
    
    // ========================================================================
    // STEP 3: Bootstrap - Generate and distribute ncclUniqueId using MPI
    // ========================================================================
    ncclUniqueId nccl_id;
    
    // Only rank 0 generates the unique ID
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }
    
    // MPI broadcasts the unique ID to all ranks
    // This is the "out-of-band" communication step
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // ========================================================================
    // STEP 4: Initialize NCCL communicator
    // ========================================================================
    ncclComm_t nccl_comm;
    
    // All ranks call ncclCommInitRank with the shared unique ID
    // NCCL will now:
    //   1. Use TCP to connect all ranks (bootstrap network)
    //   2. Exchange GPU topology information
    //   3. Set up high-performance communication channels (NVLink, GPU Direct, etc.)
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));
    
    std::cout << "[Rank " << rank << "] NCCL initialized" << std::endl;
    
    // ========================================================================
    // STEP 5: Test the communicator with AllReduce
    // ========================================================================
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(float)));
    
    // Each rank contributes its rank value
    float h_data = static_cast<float>(rank);
    CUDA_CHECK(cudaMemcpy(d_data, &h_data, sizeof(float), cudaMemcpyHostToDevice));
    
    // Perform AllReduce (sum)
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    NCCL_CHECK(ncclAllReduce(d_data, d_data, 1, ncclFloat, ncclSum, nccl_comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(&h_data, d_data, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Expected result: sum of all ranks = 0 + 1 + 2 + ... + (world_size-1)
    float expected = (world_size * (world_size - 1)) / 2.0f;
    
    if (h_data == expected) {
        std::cout << "[Rank " << rank << "] AllReduce test PASSED (result=" << h_data << ")" << std::endl;
    } else {
        std::cout << "[Rank " << rank << "] AllReduce test FAILED (expected=" << expected 
                  << ", got=" << h_data << ")" << std::endl;
    }
    
    // ========================================================================
    // STEP 6: Cleanup
    // ========================================================================
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    
    MPI_Finalize();
    return 0;
}

/*
 * COMPILE:
 *   mpic++ -o mpi_bootstrap example_mpi_bootstrap.cpp -lnccl -lcudart
 * 
 * RUN (4 processes):
 *   mpirun -np 4 ./mpi_bootstrap
 * 
 * RUN (multi-node with hostfile):
 *   mpirun -np 8 --hostfile hosts.txt ./mpi_bootstrap
 * 
 * EXPECTED OUTPUT:
 *   [Rank 0] Starting MPI bootstrap example
 *   [Rank 0] Using GPU 0
 *   [Rank 0] Generated ncclUniqueId
 *   [Rank 1] Starting MPI bootstrap example
 *   [Rank 1] Using GPU 1
 *   [Rank 1] Received ncclUniqueId from rank 0
 *   ...
 *   [Rank 0] After AllReduce: 6.0 (expected: 6.0)
 *   [Rank 0] ✓ AllReduce test PASSED
 * 
 * KEY POINTS:
 *   - MPI_Bcast is the "out-of-band" mechanism
 *   - After ncclCommInitRank, NCCL handles all GPU communication
 *   - MPI can still be used for CPU-side communication if needed
 */
