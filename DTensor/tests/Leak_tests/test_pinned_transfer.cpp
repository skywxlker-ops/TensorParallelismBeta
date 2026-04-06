#include "core/Tensor.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace OwnTensor;

class Timer {
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clock> start_time;
public:
    void start() { start_time = Clock::now(); }
    double stop() {
        auto end_time = Clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  PINNED MEMORY TRANSFER BENCHMARK" << std::endl;
    std::cout << "================================================================" << std::endl;

    // Parameters
    // 512MB transfer
    int64_t num_elements = 128 * 1024 * 1024; 
    Shape shape{{num_elements}};
    
    std::cout << "Transfer Size: " << (num_elements * 4 / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // 1. Pageable Memory (Standard)
    std::cout << "\n[1] Pageable Memory (Standard New)" << std::endl;
    Tensor t_pageable = Tensor::randn<float>(shape, TensorOptions().with_device(Device::CPU));
    // Warmup
    t_pageable.to_cuda();
    
    Timer timer;
    timer.start();
    for (int i=0; i<10; i++) {
        Tensor t_gpu = t_pageable.to_cuda();
    }
    double time_pageable = timer.stop() / 10.0;
    std::cout << "Avg Transfer Time: " << time_pageable << " ms" << std::endl;
    std::cout << "Bandwidth: " << (num_elements * 4 * 1e-6 / (time_pageable * 1e-3)) << " GB/s" << std::endl;
    
    // 2. Pinned Memory
    std::cout << "\n[2] Pinned Memory (cudaMallocHost)" << std::endl;
    
    timer.start();
    Tensor t_pinned = t_pageable.pin_memory();
    double pin_time = timer.stop();
    std::cout << "Pinning Time (Copy): " << pin_time << " ms" << std::endl;
    
    // Warmup
    t_pinned.to_cuda();
    
    timer.start();
    for (int i=0; i<10; i++) {
        Tensor t_gpu = t_pinned.to_cuda();
    }
    double time_pinned = timer.stop() / 10.0;
    std::cout << "Avg Transfer Time: " << time_pinned << " ms" << std::endl;
    std::cout << "Bandwidth: " << (num_elements * 4 * 1e-6 / (time_pinned * 1e-3)) << " GB/s" << std::endl;
    
    std::cout << "\nSpeedup: " << (time_pageable / time_pinned) << "x" << std::endl;
    
    return 0;
}
