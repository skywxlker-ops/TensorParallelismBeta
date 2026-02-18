#include "autograd_test_utils.h"

using namespace OwnTensor;

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  STANDALONE SIGMOID OPERATION BENCHMARK" << std::endl;
    std::cout << "================================================================" << std::endl;

    // Detect device
    std::cout << "Running on: CUDA (Default Device 0)" << std::endl;

    int64_t initial_active = Tensor::get_active_tensor_count();
    std::cout << "Initial Active Tensors: " << initial_active << std::endl;
    std::cout << "\n=== Sigmoid Test ===\n" << std::endl;

    Device device = Device::CUDA;
    TensorOptions opts = TensorOptions().with_device(device).with_dtype(Dtype::Float32).with_req_grad(true);
    
    // Parameters
    int64_t B = 8, T = 128, C = 768; // Same as reduced suite to avoid OOM
    Shape shape{{B*T, C}};
    
    std::cout << "Input Shape: [" << B*T << ", " << C << "] = " << (B*T*C) << " elements" << std::endl;

    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        
        // Input
        Tensor x = Tensor::randn<float>(shape, opts);
        
        // CPU Reference for correctness check
        Tensor x_cpu = x.to_cpu();
        Tensor expected = autograd::sigmoid(x_cpu);
        
        // Target Function on Device
        auto run_fn = [&]( ) { return autograd::sigmoid(x); };
        
        // Run Benchmark
        // Note: Leaks will likely be high (~2 * iterations) due to graph cycle (Output->GradFn->Output)
        TestMetrics m = benchmark_op("Sigmoid", run_fn, x, expected.to(device));
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 1; 
        
        print_result("Sigmoid", m);
    }

    std::cout << "\n================================================================" << std::endl;
    int64_t final_active = Tensor::get_active_tensor_count();
    std::cout << "Final Active Tensors: " << final_active << std::endl;
    std::cout << "Net Leak: " << (final_active - initial_active) << std::endl;

    return 0;
}
