/**
 * Test: CachingCUDAAllocator Integration
 */

#include <unparalleled/unparalleled.h>
#include <iostream>
#include <chrono>
#include <iomanip>

#include "memory/CachingCUDAAllocator.h"

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    cudaSetDevice(rank);
    
    OwnTensor::CachingCUDAAllocator allocator;
    
    // Test reuse
    size_t sz = 1024 * 1024;
    void* p1 = allocator.allocate(sz);
    allocator.deallocate(p1);
    void* p2 = allocator.allocate(sz);
    allocator.deallocate(p2);
    bool reused = (p1 == p2);
    
    // Benchmark
    auto start = Clock::now();
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; ++i) {
        ptrs.push_back(allocator.allocate(64 * 1024));
    }
    for (auto p : ptrs) allocator.deallocate(p);
    auto mid = Clock::now();
    
    ptrs.clear();
    for (int i = 0; i < 100; ++i) {
        ptrs.push_back(allocator.allocate(64 * 1024));
    }
    for (auto p : ptrs) allocator.deallocate(p);
    auto end = Clock::now();
    
    if (rank == 0) {
        std::cout << "CachingAlloc: reuse=" << (reused ? "yes" : "no")
                  << ", fresh=" << std::fixed << std::setprecision(2) << Ms(mid - start).count() << "ms"
                  << ", cached=" << Ms(end - mid).count() << "ms"
                  << ", pool=" << allocator.memoryFree() / (1024*1024) << "MB" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}

// Build: cd tests && make test_caching_allocator
// Run:   mpirun -np 2 ./test_caching_allocator
