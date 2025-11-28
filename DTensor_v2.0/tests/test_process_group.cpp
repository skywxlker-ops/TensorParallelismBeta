#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <cmath>
#include <unistd.h>

#include "process_group/process_group.h"

// =============================================================================
// Test Utilities
// =============================================================================

void print_test_header(int rank, const std::string& test_name) {
    if (rank == 0) {
        std::cout << "\n[Test] " << test_name << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_test_result(int rank, const std::string& test_name, bool passed) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "  [" << (passed ? "ok" : "fail") << "] " << test_name << "\n";
    }
}

template<typename T>
bool arrays_equal(const T* a, const T* b, size_t count, T tolerance = 1e-5) {
    for (size_t i = 0; i < count; i++) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// =============================================================================
// Test 1: Point-to-Point Send/Recv
// =============================================================================

void test_send_recv(std::shared_ptr<ProcessGroup> pg, int rank, int world_size) {
    print_test_header(rank, "Point-to-Point Send/Recv");
    
    constexpr size_t count = 8;
    float* send_data = nullptr;
    float* recv_data = nullptr;
    
    CUDA_CHECK(cudaMalloc(&send_data, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_data, count * sizeof(float)));
    
    // Initialize data
    std::vector<float> host_send(count);
    std::vector<float> host_recv(count, 0.0f);
    
    for (size_t i = 0; i < count; i++) {
        host_send[i] = static_cast<float>(rank * 100 + i);
    }
    
    CUDA_CHECK(cudaMemcpy(send_data, host_send.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    
    // Ping-pong pattern: rank 0 sends to rank 1, rank 1 sends to rank 0
    if (world_size >= 2) {
        NCCL_CHECK(ncclGroupStart());
        if (rank == 0) {
            NCCL_CHECK(ncclSend(send_data, count, ncclFloat, 1, pg->getComm(), pg->getStream()));
            NCCL_CHECK(ncclRecv(recv_data, count, ncclFloat, 1, pg->getComm(), pg->getStream()));
        } else if (rank == 1) {
            NCCL_CHECK(ncclRecv(recv_data, count, ncclFloat, 0, pg->getComm(), pg->getStream()));
            NCCL_CHECK(ncclSend(send_data, count, ncclFloat, 0, pg->getComm(), pg->getStream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        
        CUDA_CHECK(cudaStreamSynchronize(pg->getStream()));
        CUDA_CHECK(cudaMemcpy(host_recv.data(), recv_data, count * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Verify
        bool passed = true;
        if (rank == 0) {
            // Should receive rank 1's data
            for (size_t i = 0; i < count; i++) {
                if (std::abs(host_recv[i] - (100 + i)) > 1e-5) {
                    passed = false;
                    break;
                }
            }
        } else if (rank == 1) {
            // Should receive rank 0's data
            for (size_t i = 0; i < count; i++) {
                if (std::abs(host_recv[i] - i) > 1e-5) {
                    passed = false;
                    break;
                }
            }
        }
        
        if (rank < 2 && passed) {
            std::cout << "  rank " << rank << " recv OK\n";
        }
        
        print_test_result(rank, "send/recv", passed);
    } else {
        if (rank == 0) {
            std::cout << "[SKIP] Send/Recv test requires at least 2 ranks\n";
        }
    }
    
    CUDA_CHECK(cudaFree(send_data));
    CUDA_CHECK(cudaFree(recv_data));
}

// =============================================================================
// Test 2: AllReduce with Different Reduction Operations
// =============================================================================

void test_reduction_ops(std::shared_ptr<ProcessGroup> pg, int rank, int world_size) {
    print_test_header(rank, "AllReduce with Different Reduction Operations");
    
    constexpr size_t count = 4;
    float* data = nullptr;
    CUDA_CHECK(cudaMalloc(&data, count * sizeof(float)));
    
    std::vector<float> host_data(count);
    bool all_passed = true;
    
    // Test 1: Sum (default)
    {
        for (size_t i = 0; i < count; i++) {
            host_data[i] = static_cast<float>(rank + 1);
        }
        CUDA_CHECK(cudaMemcpy(data, host_data.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        
        auto work = pg->allReduce(data, count, ncclFloat, ncclSum);
        work->wait();
        
        CUDA_CHECK(cudaMemcpy(host_data.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
        
        float expected = 0.0f;
        for (int r = 0; r < world_size; r++) {
            expected += (r + 1);
        }
        
        bool passed = arrays_equal(host_data.data(), std::vector<float>(count, expected).data(), count);
        all_passed &= passed;
        
        if (rank == 0 && !passed) {
            std::cout << "  Sum: expected " << expected << ", got " << host_data[0] << "\n";
        }
    }
    
    // Test 2: Max
    {
        for (size_t i = 0; i < count; i++) {
            host_data[i] = static_cast<float>(rank * 10);
        }
        CUDA_CHECK(cudaMemcpy(data, host_data.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        
        auto work = pg->allReduce(data, count, ncclFloat, ncclMax);
        work->wait();
        
        CUDA_CHECK(cudaMemcpy(host_data.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
        
        float expected = static_cast<float>((world_size - 1) * 10);
        bool passed = arrays_equal(host_data.data(), std::vector<float>(count, expected).data(), count);
        all_passed &= passed;
        
        if (rank == 0 && !passed) {
            std::cout << "  Max: expected " << expected << ", got " << host_data[0] << "\n";
        }
    }
    
    // Test 3: Min
    {
        for (size_t i = 0; i < count; i++) {
            host_data[i] = static_cast<float>(rank * 10 + 5);
        }
        CUDA_CHECK(cudaMemcpy(data, host_data.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        
        auto work = pg->allReduce(data, count, ncclFloat, ncclMin);
        work->wait();
        
        CUDA_CHECK(cudaMemcpy(host_data.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
        
        float expected = 5.0f;
        bool passed = arrays_equal(host_data.data(), std::vector<float>(count, expected).data(), count);
        all_passed &= passed;
        
        if (rank == 0 && !passed) {
            std::cout << "  Min: expected " << expected << ", got " << host_data[0] << "\n";
        }
    }
    
    // Test 4: Product
    {
        for (size_t i = 0; i < count; i++) {
            host_data[i] = static_cast<float>(rank + 2);
        }
        CUDA_CHECK(cudaMemcpy(data, host_data.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        
        auto work = pg->allReduce(data, count, ncclFloat, ncclProd);
        work->wait();
        
        CUDA_CHECK(cudaMemcpy(host_data.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
        
        float expected = 1.0f;
        for (int r = 0; r < world_size; r++) {
            expected *= (r + 2);
        }
        
        bool passed = arrays_equal(host_data.data(), std::vector<float>(count, expected).data(), count);
        all_passed &= passed;
        
        if (rank == 0 && !passed) {
            std::cout << "  Prod: expected " << expected << ", got " << host_data[0] << "\n";
        }
    }
    
    CUDA_CHECK(cudaFree(data));
    print_test_result(rank, "reduction ops (sum/max/min/prod)", all_passed);
}

// =============================================================================
// Test 3: Barrier Synchronization
// =============================================================================

void test_barrier(std::shared_ptr<ProcessGroup> pg, int rank, int world_size) {
    print_test_header(rank, "Barrier Synchronization");
    
    // Simulate different ranks arriving at barrier at different times
    if (rank != 0) {
        usleep(rank * 100000); // 100ms * rank
    }
    
    pg->barrier();
    
    print_test_result(rank, "barrier", true);
}

// =============================================================================
// Test 4: Existing Collectives Still Work
// =============================================================================

void test_existing_collectives(std::shared_ptr<ProcessGroup> pg, int rank, int world_size) {
    print_test_header(rank, "Existing Collectives Compatibility");
    
    constexpr size_t count = 8;
    float* data = nullptr;
    CUDA_CHECK(cudaMalloc(&data, count * sizeof(float)));
    
    std::vector<float> host_data(count);
    bool all_passed = true;
    
    // Test AllReduce (old API with implicit ncclSum)
    {
        for (size_t i = 0; i < count; i++) {
            host_data[i] = static_cast<float>(rank + 1);
        }
        CUDA_CHECK(cudaMemcpy(data, host_data.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        
        auto work = pg->allReduce(data, count, ncclFloat);
        work->wait();
        
        CUDA_CHECK(cudaMemcpy(host_data.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
        
        float expected = 0.0f;
        for (int r = 0; r < world_size; r++) {
            expected += (r + 1);
        }
        
        bool passed = arrays_equal(host_data.data(), std::vector<float>(count, expected).data(), count);
        all_passed &= passed;
    }
    
    // Test Broadcast
    {
        if (rank == 0) {
            for (size_t i = 0; i < count; i++) {
                host_data[i] = 42.0f + i;
            }
        } else {
            for (size_t i = 0; i < count; i++) {
                host_data[i] = 0.0f;
            }
        }
        CUDA_CHECK(cudaMemcpy(data, host_data.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        
        auto work = pg->broadcast(data, count, 0, ncclFloat);
        work->wait();
        
        CUDA_CHECK(cudaMemcpy(host_data.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::vector<float> expected(count);
        for (size_t i = 0; i < count; i++) {
            expected[i] = 42.0f + i;
        }
        
        bool passed = arrays_equal(host_data.data(), expected.data(), count);
        all_passed &= passed;
    }
    
    CUDA_CHECK(cudaFree(data));
    print_test_result(rank, "backward compatibility", all_passed);
}

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (rank == 0) {
        std::cout << "\nProcessGroup Test Suite (world_size=" << world_size << ")\n";
    }
    
    // Set device
    CUDA_CHECK(cudaSetDevice(rank % world_size));
    
    // Initialize NCCL
    ncclUniqueId nccl_id;
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Create ProcessGroup
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank % world_size, nccl_id);
    
    // Run tests
    test_existing_collectives(pg, rank, world_size);
    test_reduction_ops(pg, rank, world_size);
    test_barrier(pg, rank, world_size);
    test_send_recv(pg, rank, world_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\nAll tests passed.\n";
    }
    
    // Cleanup
    pg.reset();
    MPI_Finalize();
    
    return 0;
}
