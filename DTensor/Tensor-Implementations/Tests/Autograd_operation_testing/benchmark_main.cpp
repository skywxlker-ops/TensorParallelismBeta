#include <iostream>
#include "autograd_test_utils.h"

// Forward declarations of test runners
void run_activation_tests();
void run_loss_tests();
void run_layer_tests();

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  AUTOGRAD OPERATION BENCHMARK & VERIFICATION" << std::endl;
    std::cout << "================================================================" << std::endl;
    
#ifdef WITH_CUDA
    std::cout << "Running on: CUDA (Default Device 0)" << std::endl;
#else
    std::cout << "Running on: CPU" << std::endl;
#endif
    std::cout << "Checking for memory leaks via Tensor Active Count..." << std::endl;
    
    int64_t initial_count = Tensor::get_active_tensor_count();
    std::cout << "Initial Active Tensors: " << initial_count << std::endl;
    
    // Run Suites
    try {
        run_activation_tests();
        run_loss_tests();
        run_layer_tests();
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR CAUGHT: " << e.what() << std::endl;
    }
    
    std::cout << "\n================================================================" << std::endl;
    int64_t final_count = Tensor::get_active_tensor_count();
    std::cout << "Final Active Tensors: " << final_count << std::endl;
    std::cout << "Net Leak: " << (final_count - initial_count) << std::endl;
    
    if (final_count > initial_count) {
        std::cout << "WARNING: Memory leaks detected! (Or static tensors persisting)" << std::endl;
    } else {
        std::cout << "Memory Efficiency: OK (No dangling tensors)" << std::endl;
    }
    
    return 0;
}
