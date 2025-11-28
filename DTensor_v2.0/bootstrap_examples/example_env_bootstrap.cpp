/*
 * NCCL Bootstrap Example 3: Environment Variable Bootstrap
 * 
 * This example demonstrates using environment variables for NCCL bootstrap.
 * The ncclUniqueId is encoded as a hex string and passed via NCCL_COMM_ID.
 * 
 * Advantages:
 *   - Cloud-native (Docker, Kubernetes)
 *   - No shared filesystem needed
 *   - Works with job schedulers (SLURM, Ray, etc.)
 * 
 * Disadvantages:
 *   - Need external coordination to generate and set env var
 *   - Env var size limits (usually 128KB+, sufficient for ncclUniqueId)
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iomanip>

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
// Helper Functions for Environment Variable Bootstrap
// ============================================================================

// Convert ncclUniqueId to hex string
std::string nccl_id_to_hex(const ncclUniqueId& id) {
    std::ostringstream oss;
    for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; i++) {
        oss << std::hex << std::setw(2) << std::setfill('0') 
            << (int)(unsigned char)id.internal[i];
    }
    return oss.str();
}

// Convert hex string to ncclUniqueId
void hex_to_nccl_id(const std::string& hex, ncclUniqueId& id) {
    if (hex.length() != NCCL_UNIQUE_ID_BYTES * 2) {
        std::cerr << "Invalid hex string length: " << hex.length() 
                  << " (expected " << NCCL_UNIQUE_ID_BYTES * 2 << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; i++) {
        std::string byte_str = hex.substr(i * 2, 2);
        id.internal[i] = (char)strtol(byte_str.c_str(), nullptr, 16);
    }
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char** argv) {
    // Parse command-line arguments
    int rank = -1;
    int world_size = -1;
    bool generate_only = false;
    
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--rank=", 7) == 0) {
            rank = atoi(argv[i] + 7);
        } else if (strncmp(argv[i], "--world-size=", 13) == 0) {
            world_size = atoi(argv[i] + 13);
        } else if (strcmp(argv[i], "--generate") == 0) {
            generate_only = true;
        }
    }
    
    // ========================================================================
    // Special mode: Generate ncclUniqueId and print as env var
    // ========================================================================
    if (generate_only) {
        ncclUniqueId id;
        NCCL_CHECK(ncclGetUniqueId(&id));
        
        std::string hex = nccl_id_to_hex(id);
        
        std::cout << "# Copy and run this command to set the environment variable:" 
                  << std::endl;
        std::cout << "export NCCL_COMM_ID=" << hex << std::endl;
        
        return 0;
    }
    
    // ========================================================================
    // Normal mode: Use NCCL_COMM_ID from environment
    // ========================================================================
    if (rank < 0 || world_size <= 0) {
        std::cerr << "Usage: " << argv[0] 
                  << " --rank=<rank> --world-size=<size>" << std::endl;
        std::cerr << "   or: " << argv[0] << " --generate" << std::endl;
        return 1;
    }
    
    std::cout << "[Rank " << rank << "] Environment-based bootstrap" << std::endl;
    
    // ========================================================================
    // STEP 1: Set CUDA device
    // ========================================================================
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    
    int device_id = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    std::cout << "[Rank " << rank << "] Using GPU " << device_id << std::endl;
    
    // ========================================================================
    // STEP 2: Bootstrap - Read from environment variable
    // ========================================================================
    const char* nccl_comm_id_env = std::getenv("NCCL_COMM_ID");
    
    if (!nccl_comm_id_env) {
        std::cerr << "[Rank " << rank 
                  << "] ERROR: NCCL_COMM_ID environment variable not set!" 
                  << std::endl;
        std::cerr << "Run: " << argv[0] << " --generate" << std::endl;
        std::cerr << "Then export the NCCL_COMM_ID before running ranks" 
                  << std::endl;
        return 1;
    }
    
    ncclUniqueId nccl_id;
    hex_to_nccl_id(std::string(nccl_comm_id_env), nccl_id);
    
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
    
    return 0;
}

/*
 * COMPILE:
 *   nvcc -o env_bootstrap example_env_bootstrap.cpp -lnccl
 * 
 * RUN STEP 1 - Generate the NCCL_COMM_ID:
 *   ./env_bootstrap --generate
 *   # Output:
 *   # export NCCL_COMM_ID=0a1b2c3d4e5f...
 * 
 * RUN STEP 2 - Export the environment variable:
 *   export NCCL_COMM_ID=0a1b2c3d4e5f...  # (paste from step 1)
 * 
 * RUN STEP 3 - Launch all ranks:
 *   ./env_bootstrap --rank=0 --world-size=4 &
 *   ./env_bootstrap --rank=1 --world-size=4 &
 *   ./env_bootstrap --rank=2 --world-size=4 &
 *   ./env_bootstrap --rank=3 --world-size=4 &
 *   wait
 * 
 * KUBERNETES EXAMPLE (deployment.yaml):
 *   apiVersion: v1
 *   kind: Pod
 *   metadata:
 *     name: nccl-worker-0
 *   spec:
 *     containers:
 *     - name: worker
 *       image: my-nccl-image
 *       env:
 *       - name: NCCL_COMM_ID
 *         valueFrom:
 *           configMapKeyRef:
 *             name: nccl-config
 *             key: comm-id
 *       - name: RANK
 *         value: "0"
 *       - name: WORLD_SIZE
 *         value: "4"
 *       command: ["/app/env_bootstrap", "--rank=$(RANK)", "--world-size=$(WORLD_SIZE)"]
 * 
 * DOCKER COMPOSE EXAMPLE:
 *   version: '3'
 *   services:
 *     worker-0:
 *       image: my-nccl-image
 *       environment:
 *         - NCCL_COMM_ID=${NCCL_COMM_ID}
 *       command: ./env_bootstrap --rank=0 --world-size=4
 *     worker-1:
 *       image: my-nccl-image
 *       environment:
 *         - NCCL_COMM_ID=${NCCL_COMM_ID}
 *       command: ./env_bootstrap --rank=1 --world-size=4
 * 
 * KEY POINTS:
 *   - Generate NCCL_COMM_ID once and share it to all containers/pods
 *   - All ranks must see the same NCCL_COMM_ID value
 *   - Works great with orchestrators (K8s, Docker Swarm, Nomad)
 *   - Can use ConfigMaps, Secrets, or environment injection
 */
