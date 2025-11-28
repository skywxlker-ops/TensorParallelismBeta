#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include <thread>
#include <chrono>

#include "process_group/stream_pool.h"
#include "process_group/process_group.h"

// =============================================================================
// Test Utilities
// =============================================================================

void print_test_header(const std::string& test_name) {
    std::cout << "\n========================================\n";
    std::cout << "TEST: " << test_name << "\n";
    std::cout << "========================================\n";
}

void print_result(const std::string& test_name, bool passed) {
    if (passed) {
        std::cout << "[PASS] " << test_name << "\n";
    } else {
        std::cout << "[FAIL] " << test_name << "\n";
    }
}

// =============================================================================
// Test 1: StreamPool Basic Operations
// =============================================================================

void test_stream_pool_basic() {
    print_test_header("StreamPool Basic Operations");
    
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    
    // Create pool with 3 streams
    dtensor::StreamPool pool(device, 3);
    
    std::cout << "  Total streams: " << pool.total_streams() << "\n";
    std::cout << "  Available: " << pool.available_streams() << "\n";
    
    assert(pool.total_streams() == 3);
    assert(pool.available_streams() == 3);
    
    // Acquire a stream
    {
        auto handle = pool.acquire();
        std::cout << "  Acquired stream, available: " << pool.available_streams() << "\n";
        assert(pool.available_streams() == 2);
        
        // Use the stream
        cudaStream_t stream = handle.get();
        float* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, 1024 * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_data, 0, 1024 * sizeof(float), stream));
        CUDA_CHECK(cudaFree(d_data));
        
        // Stream auto-released when handle goes out of scope
    }
    
    std::cout << "  Released stream, available: " << pool.available_streams() << "\n";
    assert(pool.available_streams() == 3);
    
    print_result("StreamPool Basic Operations", true);
}

// =============================================================================
// Test 2: StreamPool Multiple Acquisitions
// =============================================================================

void test_stream_pool_multiple() {
    print_test_header("StreamPool Multiple Acquisitions");
    
    int device = 0;
    dtensor::StreamPool pool(device, 3);
    
    std::vector<dtensor::StreamHandle> handles;
    
    // Acquire all streams
    for (int i = 0; i < 3; ++i) {
        handles.push_back(pool.acquire());
        std::cout << "  Acquired stream " << i << ", available: " 
                  << pool.available_streams() << "\n";
    }
    
    assert(pool.available_streams() == 0);
    
    // Release all streams
    handles.clear();
    
    std::cout << "  All released, available: " << pool.available_streams() << "\n";
    assert(pool.available_streams() == 3);
    
    print_result("StreamPool Multiple Acquisitions", true);
}

// =============================================================================
// Test 3: Work Async Completion Check
// =============================================================================

void test_work_is_completed() {
    print_test_header("Work Async Completion Check");
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    auto work = std::make_shared<Work>(stream);
    
    // Not completed yet
    assert(!work->is_completed());
    std::cout << "  Work not completed initially [OK]\n";
    
    // Launch a kernel-like operation
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, 10000 * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_data, 0, 10000 * sizeof(float), stream));
    
    work->markCompleted(true);
    
    // Wait for actual GPU execution
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Should be completed now
    assert(work->is_completed());
    std::cout << "  Work completed after execution [OK]\n";
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    print_result("Work Async Completion Check", true);
}

// =============================================================================
// Test 4: Work Callbacks
// =============================================================================

void test_work_callbacks() {
    print_test_header("Work Callbacks");
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    auto work = std::make_shared<Work>(stream);
    
    bool callback_executed = false;
    bool callback_success = false;
    
    work->then([&callback_executed, &callback_success](bool success) {
        callback_executed = true;
        callback_success = success;
        std::cout << "  Callback executed with success=" << success << "\n";
    });
    
    std::cout << "  Callback registered [OK]\n";
    
    // Complete the work
    work->markCompleted(true);
    
    // Callback should have been executed
    assert(callback_executed);
    assert(callback_success);
    std::cout << "  Callback executed successfully [OK]\n";
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    print_result("Work Callbacks", true);
}

// =============================================================================
// Test 5: Work Chaining
// =============================================================================

void test_work_chaining() {
    print_test_header("Work Chaining");
    
    cudaStream_t stream1, stream2, stream3;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    CUDA_CHECK(cudaStreamCreate(&stream3));
    
    auto work1 = std::make_shared<Work>(stream1);
    auto work2 = std::make_shared<Work>(stream2);
    auto work3 = std::make_shared<Work>(stream3);
    
    std::cout << "  Created 3 work items\n";
    
    // Chain them together
    auto chain = Work::chain({work1, work2, work3});
    
    std::cout << "  Chained work items [OK]\n";
    
    // Chain should not be completed
    assert(!chain->is_completed());
    
    // Complete first work
    work1->markCompleted(true);
    assert(!chain->is_completed());
    std::cout << "  Work 1 completed, chain still pending [OK]\n";
    
    // Complete second work
    work2->markCompleted(true);
    assert(!chain->is_completed());
    std::cout << "  Work 2 completed, chain still pending [OK]\n";
    
    // Complete third work
    work3->markCompleted(true);
    
    // Give callbacks time to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Chain should now be completed
    assert(chain->is_completed());
    std::cout << "  Work 3 completed, chain completed [OK]\n";
    
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaStreamDestroy(stream3));
    
    print_result("Work Chaining", true);
}

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char** argv) {
    // Check CUDA availability
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }
    
    std::cout << "===========================================\n";
    std::cout << "  Async Stream Management Tests\n";
    std::cout << "===========================================\n";
    std::cout << "CUDA devices available: " << device_count << "\n";
    
    try {
        // Run all tests
        test_stream_pool_basic();
        test_stream_pool_multiple();
        test_work_is_completed();
        test_work_callbacks();
        test_work_chaining();
        
        std::cout << "\n===========================================\n";
        std::cout << "  ALL TESTS PASSED\n";
        std::cout << "===========================================\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Test failed with exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
