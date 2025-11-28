/*
 * NCCL Bootstrap Example 2: File-Based Bootstrap
 * 
 * This example demonstrates using a shared file for NCCL bootstrap.
 * Rank 0 writes the ncclUniqueId to a file, and other ranks read it.
 * 
 * Advantages:
 *   - No MPI dependency
 *   - Great for debugging (can inspect the file)
 *   - Works with SLURM and other schedulers
 * 
 * Disadvantages:
 *   - Requires shared filesystem (NFS, Lustre, etc.)
 *   - Manual process management
 *   - Need to clean up temporary files
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>

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

// ============================================================================
// Helper Functions for File-Based Bootstrap
// ============================================================================

// Write ncclUniqueId to a file (called by rank 0)
void write_nccl_id(const char* filepath, const ncclUniqueId& id) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to create file: " << filepath << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    file.write(reinterpret_cast<const char*>(&id), sizeof(ncclUniqueId));
    file.close();
    
    std::cout << "[Rank 0] Wrote ncclUniqueId to " << filepath << std::endl;
}

// Read ncclUniqueId from a file (called by all ranks)
void read_nccl_id(const char* filepath, ncclUniqueId& id, int rank) {
    // Wait for file to exist (polling with timeout)
    int timeout_seconds = 30;
    int elapsed = 0;
    
    while (access(filepath, F_OK) != 0) {
        if (elapsed >= timeout_seconds) {
            std::cerr << "[Rank " << rank << "] Timeout waiting for " 
                      << filepath << std::endl;
            std::exit(EXIT_FAILURE);
        }
        usleep(100000);  // Sleep 100ms
        elapsed++;
    }
    
    // File exists, now read it
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "[Rank " << rank << "] Failed to open file: " 
                  << filepath << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    file.read(reinterpret_cast<char*>(&id), sizeof(ncclUniqueId));
    file.close();
    
    std::cout << "[Rank " << rank << "] Read ncclUniqueId from " 
              << filepath << std::endl;
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char** argv) {
    // Parse command-line arguments
    int rank = -1;
    int world_size = -1;
    const char* shared_file = "/tmp/nccl_id.bin";  // Default path
    
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--rank=", 7) == 0) {
            rank = atoi(argv[i] + 7);
        } else if (strncmp(argv[i], "--world-size=", 13) == 0) {
            world_size = atoi(argv[i] + 13);
        } else if (strncmp(argv[i], "--file=", 7) == 0) {
            shared_file = argv[i] + 7;
        }
    }
    
    if (rank < 0 || world_size <= 0) {
        std::cerr << "Usage: " << argv[0] 
                  << " --rank=<rank> --world-size=<size> [--file=<path>]" 
                  << std::endl;
        return 1;
    }
    
    std::cout << "[Rank " << rank << "] File-based bootstrap (world_size=" << world_size << ")" << std::endl;
    
    // ========================================================================
    // STEP 1: Set CUDA device
    // ========================================================================
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    
    int device_id = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    std::cout << "[Rank " << rank << "] Using GPU " << device_id << std::endl;
    
    // ========================================================================
    // STEP 2: Bootstrap - File-based ID distribution
    // ========================================================================
    ncclUniqueId nccl_id;
    
    if (rank == 0) {
        // Rank 0: Generate and write the unique ID
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        write_nccl_id(shared_file, nccl_id);
    } else {
        // Other ranks: Wait for and read the unique ID
        read_nccl_id(shared_file, nccl_id, rank);
    }
    
    // ========================================================================
    // STEP 3: Initialize NCCL communicator
    // ========================================================================
    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));
    std::cout << "[Rank " << rank << "] NCCL initialized" << std::endl;
    
    // ========================================================================
    // STEP 4: Test with AllReduce
    // ========================================================================
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(float)));
    
    float h_data = static_cast<float>(rank);
    CUDA_CHECK(cudaMemcpy(d_data, &h_data, sizeof(float), cudaMemcpyHostToDevice));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    NCCL_CHECK(ncclAllReduce(d_data, d_data, 1, ncclFloat, ncclSum, 
                              nccl_comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaMemcpy(&h_data, d_data, sizeof(float), cudaMemcpyDeviceToHost));
    
    float expected = (world_size * (world_size - 1)) / 2.0f;
    
    if (h_data == expected) {
        std::cout << "[Rank " << rank << "] Test PASSED (result=" << h_data << ")" << std::endl;
    } else {
        std::cout << "[Rank " << rank << "] Test FAILED (expected=" << expected 
                  << ", got=" << h_data << ")" << std::endl;
    }
    
    // ========================================================================
    // STEP 5: Cleanup
    // ========================================================================
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    
    // Rank 0 removes the temporary file
    if (rank == 0) {
        unlink(shared_file);
    }
    
    return 0;
}

/*
 * COMPILE:
 *   nvcc -o file_bootstrap example_file_bootstrap.cpp -lnccl
 * 
 * RUN (4 separate terminals):
 *   Terminal 1: ./file_bootstrap --rank=0 --world-size=4
 *   Terminal 2: ./file_bootstrap --rank=1 --world-size=4
 *   Terminal 3: ./file_bootstrap --rank=2 --world-size=4
 *   Terminal 4: ./file_bootstrap --rank=3 --world-size=4
 * 
 * RUN (with custom file location):
 *   ./file_bootstrap --rank=0 --world-size=4 --file=/shared/nccl_id.bin
 * 
 * RUN (with bash script for convenience):
 *   for rank in {0..3}; do
 *     ./file_bootstrap --rank=$rank --world-size=4 &
 *   done
 *   wait
 * 
 * KEY POINTS:
 *   - Rank 0 MUST start first (or at least before others timeout)
 *   - All ranks must use the same --world-size
 *   - File path must be accessible to all ranks (shared filesystem)
 *   - Good for debugging: you can inspect /tmp/nccl_id.bin
 *   - Consider using a unique file name to avoid conflicts:
 *     /tmp/nccl_id_${JOB_ID}.bin
 */
