/**
 * Shard Benchmarking with Varying Sizes
 * Loops through different tensor dimensions (C and F) with fixed batch size (B).
 */

#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <string>
#include <nccl.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <iomanip>

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void run_benchmark(int rank, std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg, 
                   int64_t B, int64_t C, int64_t F, cudaStream_t comm_stream) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float duration;
    
    // Allocate tensors BEFORE timing starts
    // w1_layout: [B, C, F] sharded on dim 1 (C) -> [B, C/2, F] per rank (if 2 GPUs)
    Layout w1_layout(*mesh, { B , C , F }, 1);
    DTensor W1(*mesh, pg, w1_layout);
    
    // Only root initializes with random data
    if (rank == 0) {
        W1.rand();
    }
    
    // Target layout for the operation
    Layout W1_asS_layout(*mesh, { B , C/2 , F });
    // Note: In the original benchmark, W1_Shard was created but not used as source/dest in shard_fused_transpose directly?
    // Wait, the original code used: W1_Shard.shard_fused_transpose( 1 , 0 , W1 );
    // So W1_Shard is the destination.
    // Create destination tensor
    DTensor W1_Shard(*mesh, pg, W1_asS_layout);
    
    // Warmup run (not timed)
    for (int i = 0; i < 10; i++ ){
        W1_Shard.shard_fused_transpose( 1 , 0 , W1 );
    } 

    cudaDeviceSynchronize();
    cudaEventRecord(start, comm_stream);

    for (int i = 0; i < 1000; i++ ){
        W1_Shard.shard_fused_transpose( 1 , 0 , W1 );
    }
    
    cudaEventRecord(stop, comm_stream);
    
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    cudaEventElapsedTime(&duration, start, stop);

    duration /= 1000; // Average per iteration

    cudaDeviceSynchronize();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int64_t size_mb = ( B * C * F * 4 ) / ( 1024 * 1024 );
    float throughput = size_mb / (duration / 1000.0f); // MB/s

    if(rank == 0){
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Shape: (" << B << ", " << C << ", " << F << ") | "
                  << "Size: " << size_mb << " MB | "
                  << "Time: " << duration << " ms | "
                  << "Throughput: " << throughput << " MB/s" << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   
    // Set CUDA device FIRST before any CUDA/NCCL operations
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        CUDA_CHECK(cudaSetDevice(rank % num_devices));
    }

    // Initialize Device Mesh and Process Group once
    std::shared_ptr<DeviceMesh> mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    std::shared_ptr<ProcessGroupNCCL> pg = init_process_group(world_size, rank);

    cudaStream_t comm_stream;
    cudaStreamCreate(&comm_stream);

    if (rank == 0) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "DTensor Native Sharding Benchmark (Varying Sizes)" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "World Size: " << world_size << std::endl;
        std::cout << "Fixed Batch (B): 8" << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;
    }

    // Varying C (channels) and F (features)
    // Note: In Python we varied t and c. Here F maps to t, C maps to c.
    std::vector<int64_t> F_values = {128, 512, 1024, 2048};
    std::vector<int64_t> C_values = {768, 1024, 2048, 4096};
    const int64_t B = 8;
    
    for (int64_t F : F_values) {
        for (int64_t C : C_values) {
            run_benchmark(rank, mesh, pg, B, C, F, comm_stream);
        }
    }

    if (rank == 0) {
        std::cout << "============================================================" << std::endl;
    }

    cudaStreamDestroy(comm_stream);
    MPI_Finalize();
    return 0;
}
