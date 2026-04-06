#include "autograd_test_utils.h"
#include "ops/helpers/ActivationKernels.h"

// Declaration
void run_activation_tests();

void run_activation_tests() {
    std::cout << "\n=== Activation Tests ===\n" << std::endl;
    
    int64_t B = 8, T = 128, C = 768;
    Shape shape{{B*T, C}};
    
    // Check GPU vs CPU
    bool use_cuda = true; // Hardcoded for verifying CUDA fused kernels
#ifndef WITH_CUDA
    use_cuda = false;
#endif

    DeviceIndex device = use_cuda ? DeviceIndex(Device::CUDA, 0) : DeviceIndex(Device::CPU);
    TensorOptions opts = TensorOptions().with_device(device).with_dtype(Dtype::Float32).with_req_grad(true);

    // --- ReLU ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        Tensor x = Tensor::randn<float>(shape, opts);
        
        // Manual CPU reference (small slice check logic or assume correctness if we trust CPU impl)
        // For rigorous check, we'd actally implement reference math here.
        // For now, we trust the kernel works if it runs without error and produces valid numbers.
        // But benchmarking requires comparison.
        Tensor x_cpu = x.to_cpu();
        Tensor expected = autograd::relu(x_cpu); 
        
        auto run_fn = [&]( ) { return autograd::relu(x); };
        
        TestMetrics m = benchmark_op("ReLU", run_fn, x, expected.to(device));
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 1; // -1 for x
        print_result("ReLU", m);
        
        x.release(); // Manually cleanup to help counter check? scoped is better.
    }

    // --- GELU ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        Tensor x = Tensor::randn<float>(shape, opts);
        Tensor x_cpu = x.to_cpu();
        Tensor expected = autograd::gelu(x_cpu);
        
        auto run_fn = [&]( ) { return autograd::gelu(x); };
        
        TestMetrics m = benchmark_op("GELU", run_fn, x, expected.to(device));
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 1;
        print_result("GELU", m);
    }
    
    // --- Sigmoid ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        Tensor x = Tensor::randn<float>(shape, opts);
        Tensor x_cpu = x.to_cpu();
        Tensor expected = autograd::sigmoid(x_cpu);
        
        auto run_fn = [&]( ) { return autograd::sigmoid(x); };
        
        TestMetrics m = benchmark_op("Sigmoid", run_fn, x, expected.to(device));
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 1; 
        print_result("Sigmoid", m);
    }
    
    // --- Softmax ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        // Use smaller shape for Softmax as it is exp-intensive
        Shape s_shape{{B*T, 1024}}; // like attn scores
        Tensor x = Tensor::randn<float>(s_shape, opts);
        
        // Reference: Softmax on CPU (last dim)
        // Note: CPU implementation might be slow, so reference check is good.
        Tensor x_cpu = x.to_cpu();
        Tensor expected = autograd::softmax(x_cpu, -1);
        
        auto run_fn = [&]( ) { return autograd::softmax(x, -1); };
        
        TestMetrics m = benchmark_op("Softmax", run_fn, x, expected.to(device));
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 1;
        print_result("Softmax", m);
    }
}
