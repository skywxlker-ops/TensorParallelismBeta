/**
 * @file test_process_group_nccl.cpp
 * @brief Standalone test file for the new ProcessGroupNCCL implementation
 * 
 * Tests all NCCL collective operations:
 * - AllReduce
 * - Broadcast  
 * - AllGather
 * - ReduceScatter
 * - Gather
 * - Scatter
 * - Reduce
 * - AllToAll
 * - SendRecv (point-to-point)
 * 
 * Run with: mpirun -np 2 ./test_process_group_nccl
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <memory>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

// Include the new ProcessGroupNCCL header
#include "../process_group/ProcessGroupNCCL.h"

// Helper to verify results on host
template<typename T>
bool verify_buffer(T* device_ptr, const std::vector<T>& expected, size_t count, const std::string& test_name, int rank) {
    std::vector<T> host_data(count);
    cudaMemcpy(host_data.data(), device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    
    bool passed = true;
    for (size_t i = 0; i < count; ++i) {
        if (std::abs(host_data[i] - expected[i]) > 1e-5) {
            if (rank == 0) {
                std::cerr << "[" << test_name << "] Mismatch at index " << i 
                          << ": got " << host_data[i] << ", expected " << expected[i] << std::endl;
            }
            passed = false;
            break;
        }
    }
    return passed;
}

// Helper to print test result
void print_result(const std::string& test_name, bool passed, int rank) {
    if (rank == 0) {
        std::cout << std::setw(25) << std::left << test_name 
                  << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (rank == 0) {
            std::cerr << "This test requires at least 2 ranks. Run with: mpirun -np 2 ./test_process_group_nccl" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Set local device
    int local_rank = rank % 8; // Assuming max 8 GPUs per node
    cudaSetDevice(local_rank);

    // Initialize ProcessGroup using the new API
    auto pg = init_process_group(world_size, rank);

    if (rank == 0) {
        std::cout << "============================================" << std::endl;
        std::cout << "  ProcessGroupNCCL Test " << std::endl;
        std::cout << "  World Size: " << world_size << std::endl;
        std::cout << "============================================" << std::endl << std::endl;
    }

    const size_t count = 1024;
    int total_tests = 0;
    int passed_tests = 0;

    // ================== Test 1: AllReduce ==================
    {
        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, count * sizeof(float));
        cudaMalloc(&recv_buf, count * sizeof(float));

        // Initialize: each rank sets all values to (rank + 1)
        std::vector<float> init_data(count, static_cast<float>(rank + 1));
        cudaMemcpy(send_buf, init_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        // AllReduce with sum
        result_t result = pg->all_reduce(send_buf, recv_buf, count, OwnTensor::Dtype::Float32, sum, true);

        // Expected: sum of 1 + 2 + ... + world_size = world_size * (world_size + 1) / 2
        float expected_val = static_cast<float>(world_size * (world_size + 1)) / 2.0f;
        std::vector<float> expected(count, expected_val);
        
        bool passed = (result == pgSuccess) && verify_buffer(recv_buf, expected, count, "AllReduce", rank);
        print_result("AllReduce (sum)", passed, rank);
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Test 2: Broadcast ==================
    {
        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, count * sizeof(float));
        cudaMalloc(&recv_buf, count * sizeof(float));

        // Root (rank 0) sets values to 42.0f
        std::vector<float> init_data(count, (rank == 0) ? 42.0f : 0.0f);
        cudaMemcpy(send_buf, init_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        result_t result = pg->broadcast(send_buf, recv_buf, count, OwnTensor::Dtype::Float32, 0, true);

        std::vector<float> expected(count, 42.0f);
        bool passed = (result == pgSuccess) && verify_buffer(recv_buf, expected, count, "Broadcast", rank);
        print_result("Broadcast", passed, rank);
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Test 3: AllGather ==================
    {
        size_t local_count = count / world_size;
        size_t global_count = count;
        
        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, global_count * sizeof(float)); // Full buffer for all_gather send positioning
        cudaMalloc(&recv_buf, global_count * sizeof(float));

        // Initialize full buffer with zeros, then set this rank's portion
        std::vector<float> init_data(global_count, 0.0f);
        for (size_t i = rank * local_count; i < (rank + 1) * local_count; ++i) {
            init_data[i] = static_cast<float>(rank * 100 + (i % local_count));
        }
        cudaMemcpy(send_buf, init_data.data(), global_count * sizeof(float), cudaMemcpyHostToDevice);

        result_t result = pg->all_gather(send_buf, recv_buf, local_count, OwnTensor::Dtype::Float32, true);

        // Expected: each segment contains rank's data
        std::vector<float> expected(global_count);
        for (int r = 0; r < world_size; ++r) {
            for (size_t i = 0; i < local_count; ++i) {
                expected[r * local_count + i] = static_cast<float>(r * 100 + i);
            }
        }

        bool passed = (result == pgSuccess) && verify_buffer(recv_buf, expected, global_count, "AllGather", rank);
        print_result("AllGather", passed, rank);
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Test 4: ReduceScatter ==================
    {
        size_t local_count = count / world_size;
        size_t global_count = count;

        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, global_count * sizeof(float));
        cudaMalloc(&recv_buf, local_count * sizeof(float));

        // Initialize: each rank sets identical pattern for testing reduction
        std::vector<float> init_data(global_count);
        for (size_t i = 0; i < global_count; ++i) {
            init_data[i] = static_cast<float>(i) * (rank + 1);
        }
        cudaMemcpy(send_buf, init_data.data(), global_count * sizeof(float), cudaMemcpyHostToDevice);

        result_t result = pg->reduce_scatter(send_buf, recv_buf, local_count, OwnTensor::Dtype::Float32, sum, true);

        // Expected: each rank receives sum of corresponding segments from all ranks
        float rank_sum = static_cast<float>(world_size * (world_size + 1)) / 2.0f;
        std::vector<float> expected(local_count);
        for (size_t i = 0; i < local_count; ++i) {
            expected[i] = static_cast<float>(rank * local_count + i) * rank_sum;
        }

        bool passed = (result == pgSuccess) && verify_buffer(recv_buf, expected, local_count, "ReduceScatter", rank);
        print_result("ReduceScatter (sum)", passed, rank);
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Test 5: Gather (to root) ==================
    {
        size_t local_count = count / world_size;
        
        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, local_count * sizeof(float));
        cudaMalloc(&recv_buf, count * sizeof(float));

        // Each rank sends its rank number repeated
        std::vector<float> init_data(local_count, static_cast<float>(rank));
        cudaMemcpy(send_buf, init_data.data(), local_count * sizeof(float), cudaMemcpyHostToDevice);

        result_t result = pg->gather(send_buf, recv_buf, local_count, OwnTensor::Dtype::Float32, 0, true);

        bool passed = (result == pgSuccess);
        if (rank == 0) {
            // Only root has valid gathered data
            std::vector<float> expected(count);
            for (int r = 0; r < world_size; ++r) {
                for (size_t i = 0; i < local_count; ++i) {
                    expected[r * local_count + i] = static_cast<float>(r);
                }
            }
            passed = passed && verify_buffer(recv_buf, expected, count, "Gather", rank);
        }
        
        print_result("Gather (root=0)", passed, rank);
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Test 6: Scatter (from root) ==================
    {
        size_t local_count = count / world_size;
        
        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, count * sizeof(float));
        cudaMalloc(&recv_buf, local_count * sizeof(float));

        // Root initializes data with pattern
        if (rank == 0) {
            std::vector<float> init_data(count);
            for (size_t i = 0; i < count; ++i) {
                init_data[i] = static_cast<float>(i / local_count * 100 + i % local_count);
            }
            cudaMemcpy(send_buf, init_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        }

        result_t result = pg->scatter(send_buf, recv_buf, local_count, OwnTensor::Dtype::Float32, 0, true);

        // Each rank should receive rank*100 + offset pattern
        std::vector<float> expected(local_count);
        for (size_t i = 0; i < local_count; ++i) {
            expected[i] = static_cast<float>(rank * 100 + i);
        }

        bool passed = (result == pgSuccess) && verify_buffer(recv_buf, expected, local_count, "Scatter", rank);
        print_result("Scatter (root=0)", passed, rank);
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Test 7: Reduce (to root) ==================
    {
        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, count * sizeof(float));
        cudaMalloc(&recv_buf, count * sizeof(float));

        // Each rank sets all values to (rank + 1)
        std::vector<float> init_data(count, static_cast<float>(rank + 1));
        cudaMemcpy(send_buf, init_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        result_t result = pg->reduce(send_buf, recv_buf, count, OwnTensor::Dtype::Float32, sum, 0, true);

        bool passed = (result == pgSuccess);
        if (rank == 0) {
            // Only root has valid reduced result
            float expected_val = static_cast<float>(world_size * (world_size + 1)) / 2.0f;
            std::vector<float> expected(count, expected_val);
            passed = passed && verify_buffer(recv_buf, expected, count, "Reduce", rank);
        }
        
        print_result("Reduce (sum, root=0)", passed, rank);
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Test 8: AllToAll ==================
    {
        size_t chunk_size = count / world_size;
        
        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, count * sizeof(float));
        cudaMalloc(&recv_buf, count * sizeof(float));

        // Each rank sends different data to each destination
        std::vector<float> init_data(count);
        for (int dest = 0; dest < world_size; ++dest) {
            for (size_t i = 0; i < chunk_size; ++i) {
                init_data[dest * chunk_size + i] = static_cast<float>(rank * 1000 + dest * 100 + i);
            }
        }
        cudaMemcpy(send_buf, init_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        result_t result = pg->alltoall(send_buf, recv_buf, chunk_size, OwnTensor::Dtype::Float32, true);

        // Expected: receive from each source rank the chunk meant for this rank
        std::vector<float> expected(count);
        for (int src = 0; src < world_size; ++src) {
            for (size_t i = 0; i < chunk_size; ++i) {
                expected[src * chunk_size + i] = static_cast<float>(src * 1000 + rank * 100 + i);
            }
        }

        bool passed = (result == pgSuccess) && verify_buffer(recv_buf, expected, count, "AllToAll", rank);
        print_result("AllToAll", passed, rank);
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Test 9: SendRecv (ring pattern) ==================
    if (world_size >= 2) {
        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, count * sizeof(float));
        cudaMalloc(&recv_buf, count * sizeof(float));

        // Each rank sends to next rank and receives from previous rank (ring)
        int send_to = (rank + 1) % world_size;
        int recv_from = (rank - 1 + world_size) % world_size;

        std::vector<float> init_data(count, static_cast<float>(rank * 100));
        cudaMemcpy(send_buf, init_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        result_t result = pg->sendrecv(send_buf, recv_buf, send_to, recv_from, count, OwnTensor::Dtype::Float32, true);

        std::vector<float> expected(count, static_cast<float>(recv_from * 100));
        bool passed = (result == pgSuccess) && verify_buffer(recv_buf, expected, count, "SendRecv", rank);
        print_result("SendRecv (ring)", passed, rank);
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Test 10: Async AllReduce ==================
    {
        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, count * sizeof(float));
        cudaMalloc(&recv_buf, count * sizeof(float));

        std::vector<float> init_data(count, static_cast<float>(rank + 1));
        cudaMemcpy(send_buf, init_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        // Use async version and wait manually
        auto work = pg->all_reduce_async(send_buf, recv_buf, count, OwnTensor::Dtype::Float32, sum, false);
        
        // Wait for completion
        bool wait_ok = work->wait();

        float expected_val = static_cast<float>(world_size * (world_size + 1)) / 2.0f;
        std::vector<float> expected(count, expected_val);
        
        bool passed = wait_ok && work->is_success() && verify_buffer(recv_buf, expected, count, "Async AllReduce", rank);
        print_result("AllReduce (async)", passed, rank);
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Test 11: Work Object Query ==================
    {
        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, count * sizeof(float));
        cudaMalloc(&recv_buf, count * sizeof(float));

        std::vector<float> init_data(count, static_cast<float>(rank + 1));
        cudaMemcpy(send_buf, init_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        auto work = pg->all_reduce_async(send_buf, recv_buf, count, OwnTensor::Dtype::Float32, sum, false);
        
        // Poll with query until complete
        int iterations = 0;
        while (!work->query() && iterations < 10000) {
            iterations++;
        }
        
        bool passed = work->is_success();
        print_result("Work Query Poll", passed, rank);
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Test 12: Timing API ==================
    {
        float* send_buf;
        float* recv_buf;
        cudaMalloc(&send_buf, count * sizeof(float));
        cudaMalloc(&recv_buf, count * sizeof(float));

        std::vector<float> init_data(count, 1.0f);
        cudaMemcpy(send_buf, init_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        pg->start_time();
        pg->all_reduce(send_buf, recv_buf, count, OwnTensor::Dtype::Float32, sum, true);
        float elapsed_ms;
        pg->end_time(elapsed_ms);

        bool passed = (elapsed_ms >= 0.0f);  // Just check timing works
        if (rank == 0) {
            std::cout << std::setw(25) << std::left << "Timing API" 
                      << ": " << (passed ? "PASSED" : "FAILED")
                      << " (" << elapsed_ms << " ms)" << std::endl;
        }
        total_tests++;
        if (passed) passed_tests++;

        cudaFree(send_buf);
        cudaFree(recv_buf);
    }

    // ================== Summary ==================
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "  Results: " << passed_tests << "/" << total_tests << " tests passed" << std::endl;
        if (passed_tests == total_tests) {
            std::cout << "  All tests PASSED" << std::endl;
        } else {
            std::cout << "  Some tests FAILED" << std::endl;
        }
        std::cout << "============================================" << std::endl;
    }

    MPI_Finalize();
    return (passed_tests == total_tests) ? 0 : 1;
}
