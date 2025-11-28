/*
 * NCCL Bootstrap Example 4: Single-Node Multi-GPU
 * 
 * This example demonstrates the simplest bootstrap case: all GPUs on one machine.
 * NCCL automatically uses shared memory for bootstrap, which is much faster than TCP.
 * 
 * Advantages:
 *   - Simplest setup
 *   - No network configuration needed
 *   - Fastest bootstrap time
 *   - Automatic NVLink detection
 * 
 * Disadvantages:
 *   - Only works on single node
 *   - Can't scale beyond one machine
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <omp.h>

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
    // Configuration
    // ========================================================================
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    
    if (num_gpus == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    // Allow overriding number of GPUs from command line
    if (argc > 1) {
        num_gpus = std::min(atoi(argv[1]), num_gpus);
    }
    
    std::cout << "Single-Node Multi-GPU NCCL Bootstrap" << std::endl;
    std::cout << "Using " << num_gpus << " GPU(s)" << std::endl;
    
    // Print GPU information
    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "GPU " << i << ": " << prop.name 
                  << " (" << prop.totalGlobalMem / (1024*1024*1024) << " GB)" 
                  << std::endl;
    }
    
    // ========================================================================
    // STEP 1: Generate ncclUniqueId (single-threaded)
    // ========================================================================
    ncclUniqueId nccl_id;
    NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    
    // ========================================================================
    // STEP 2: Initialize NCCL communicators (one per GPU)
    // ========================================================================
    std::vector<ncclComm_t> comms(num_gpus);
    std::vector<cudaStream_t> streams(num_gpus);
    std::vector<float*> d_data(num_gpus);
    
    std::cout << "Initializing communicators..." << std::endl;
    
    // Use OpenMP to parallelize initialization across GPUs
    #pragma omp parallel for num_threads(num_gpus)
    for (int i = 0; i < num_gpus; i++) {
        // Set device for this thread
        CUDA_CHECK(cudaSetDevice(i));
        
        // Create stream
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        
        // Initialize NCCL communicator
        // All threads use the SAME ncclUniqueId
        // NCCL detects they're on the same node and uses shared memory
        NCCL_CHECK(ncclCommInitRank(&comms[i], num_gpus, nccl_id, i));
        
        // Allocate test data
        CUDA_CHECK(cudaMalloc(&d_data[i], sizeof(float)));
    }
    
    std::cout << "Initialization complete" << std::endl;
    
    // ========================================================================
    // STEP 3: Test with AllReduce
    // ========================================================================
    std::cout << "Running AllReduce test..." << std::endl;
    
    // Each GPU initializes its data with its rank
    #pragma omp parallel for num_threads(num_gpus)
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        float h_value = static_cast<float>(i);
        CUDA_CHECK(cudaMemcpy(d_data[i], &h_value, sizeof(float), 
                              cudaMemcpyHostToDevice));
    }
    
    // Perform AllReduce in a group (can overlap operations)
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        NCCL_CHECK(ncclAllReduce(d_data[i], d_data[i], 1, ncclFloat, ncclSum, 
                                  comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());
    
    // Wait for all operations to complete
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // Verify results
    float expected = (num_gpus * (num_gpus - 1)) / 2.0f;
    bool all_passed = true;
    
    std::cout << "\nAllReduce results:" << std::endl;
    for (int i = 0; i < num_gpus; i++) {
        float h_result;
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpy(&h_result, d_data[i], sizeof(float), 
                              cudaMemcpyDeviceToHost));
        
        bool passed = (h_result == expected);
        all_passed &= passed;
        
        std::cout << "GPU " << i << ": result=" << h_result 
                  << ", expected=" << expected 
                  << " [" << (passed ? "PASS" : "FAIL") << "]" << std::endl;
    }
    
    // ========================================================================
    // STEP 4: Performance Test (Optional)
    // ========================================================================
    std::cout << "\nPerformance test (1MB AllReduce x 100):" << std::endl;
    
    const size_t test_size = 1024 * 1024 / sizeof(float);  // 1MB
    std::vector<float*> d_perf_data(num_gpus);
    
    #pragma omp parallel for num_threads(num_gpus)
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_perf_data[i], test_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_perf_data[i], 0, test_size * sizeof(float)));
    }
    
    // Warmup
    for (int iter = 0; iter < 10; iter++) {
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            NCCL_CHECK(ncclAllReduce(d_perf_data[i], d_perf_data[i], test_size, 
                                      ncclFloat, ncclSum, comms[i], streams[i]));
        }
        NCCL_CHECK(ncclGroupEnd());
    }
    
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start, streams[0]));
    
    for (int iter = 0; iter < 100; iter++) {
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            NCCL_CHECK(ncclAllReduce(d_perf_data[i], d_perf_data[i], test_size, 
                                      ncclFloat, ncclSum, comms[i], streams[i]));
        }
        NCCL_CHECK(ncclGroupEnd());
    }
    
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    CUDA_CHECK(cudaEventRecord(stop, streams[0]));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    float avg_time_ms = elapsed_ms / 100.0f;
    float bandwidth_gbps = (test_size * sizeof(float) * num_gpus) / (avg_time_ms * 1e6);
    
    std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
    
    // Cleanup performance test
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaFree(d_perf_data[i]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // ========================================================================
    // STEP 5: Cleanup
    // ========================================================================
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaFree(d_data[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        NCCL_CHECK(ncclCommDestroy(comms[i]));
    }
    
    if (all_passed) {
        std::cout << "\nTest PASSED" << std::endl;
    } else {
        std::cout << "\nTest FAILED" << std::endl;
    }
    
    return all_passed ? 0 : 1;
}

/*
 * COMPILE:
 *   nvcc -o single_node example_single_node.cpp -lnccl -Xcompiler -fopenmp
 * 
 * RUN (use all available GPUs):
 *   ./single_node
 * 
 * RUN (use specific number of GPUs):
 *   ./single_node 4
 * 
 * EXPECTED OUTPUT:
 *   ╔════════════════════════════════════════════════╗
 *   ║  Single-Node Multi-GPU NCCL Bootstrap          ║
 *   ╚════════════════════════════════════════════════╝
 *   
 *   Using 8 GPU(s)
 *   
 *     GPU 0: NVIDIA A100-SXM4-40GB (40 GB)
 *     GPU 1: NVIDIA A100-SXM4-40GB (40 GB)
 *     ...
 *   
 *   ✓ Generated ncclUniqueId
 *   Initializing NCCL communicators...
 *     ✓ GPU 0: Communicator initialized
 *     ✓ GPU 1: Communicator initialized
 *     ...
 *   
 *   Results:
 *     GPU 0: 28.0 (expected: 28.0) ✓
 *     GPU 1: 28.0 (expected: 28.0) ✓
 *     ...
 * 
 * KEY POINTS:
 *   - Simplest NCCL setup - everything on one node
 *   - NCCL automatically uses shared memory for bootstrap (faster than TCP)
 *   - NCCL automatically detects NVLink connectivity
 *   - ncclGroupStart/ncclGroupEnd allows overlapping operations
 *   - Perfect for single-machine training and development
 */
