/**
 * Test: Memory Pool Efficiency Benchmark
 */

#include <unparalleled/unparalleled.h>
#include <iostream>
#include <chrono>
#include <iomanip>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    cudaSetDevice(rank);
    
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    auto pg = init_process_group(world_size, rank);
    
    // Tensor creation benchmark
    std::vector<int64_t> shape = {1024, 1024};
    Layout layout(*mesh, shape, 0);
    
    auto start = Clock::now();
    for (int i = 0; i < 10; ++i) {
        auto t = DTensor::zeros(shape, mesh, pg, layout);
    }
    auto end = Clock::now();
    
    // Arithmetic benchmark
    shape = {512, 512};
    Layout l2(*mesh, shape, 0);
    auto a = DTensor::rand(shape, mesh, pg, l2);
    auto b = DTensor::rand(shape, mesh, pg, l2);
    
    auto start2 = Clock::now();
    for (int i = 0; i < 100; ++i) {
        auto c = a.add(b);
        auto d = c.mul(a);
    }
    auto end2 = Clock::now();
    
    if (rank == 0) {
        auto stats = OwnTensor::CachingCUDAAllocator::instance().get_stats();
        std::cout << "Memory Pool: create=" << std::fixed << std::setprecision(1) 
                  << Ms(end - start).count() << "ms, arith=" 
                  << Ms(end2 - start2).count() << "ms, "
                  << "pool=" << stats.cached / (1024*1024) << "MB" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}

// Build: cd tests && make test_memory_pool_efficiency
// Run:   mpirun -np 2 ./test_memory_pool_efficiency
