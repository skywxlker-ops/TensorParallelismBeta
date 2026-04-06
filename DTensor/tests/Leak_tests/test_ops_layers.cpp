#include "autograd_test_utils.h"
#include "autograd/operations/EmbeddingOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/NormalizationOps.h"

void run_layer_tests();

void run_layer_tests() {
    std::cout << "\n=== Layer Tests ===\n" << std::endl;
    
    int64_t B = 8, T = 128, C = 768;
    
    bool use_cuda = true;
#ifndef WITH_CUDA
    use_cuda = false;
#endif
    DeviceIndex device = use_cuda ? DeviceIndex(Device::CUDA, 0) : DeviceIndex(Device::CPU);
    TensorOptions opts = TensorOptions().with_device(device).with_dtype(Dtype::Float32).with_req_grad(true);

    // --- MatMul ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        Shape sa{{B*T, C}};
        Shape sb{{C, C*4}};
        
        Tensor a = Tensor::randn<float>(sa, opts);
        Tensor b = Tensor::randn<float>(sb, opts);
        
        Tensor a_cpu = a.to_cpu();
        Tensor b_cpu = b.to_cpu();
        Tensor expected = autograd::matmul(a_cpu, b_cpu);
        
        auto run_fn = [&]( ) { return autograd::matmul(a, b); };
        
        TestMetrics m = benchmark_op("MatMul [N,C]x[C,4C]", run_fn, a, expected.to(device), 20, 0.05f); // Fewer iters for big matmul
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 2;
        print_result("MatMul", m);
    }
    
    // --- LayerNorm ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        Shape shape{{B*T, C}};
        Shape norm_shape{{C}};
        
        Tensor x = Tensor::randn<float>(shape, opts);
        Tensor w = Tensor::ones(norm_shape, opts);
        Tensor b = Tensor::zeros(norm_shape, opts);
        
        Tensor x_cpu = x.to_cpu();
        Tensor w_cpu = w.to_cpu();
        Tensor b_cpu = b.to_cpu();
        int wrapped_norm_shape = norm_shape.dims.back();
        Tensor expected = autograd::layer_norm(x_cpu, w_cpu, b_cpu, wrapped_norm_shape);
        
        auto run_fn = [&]( ) { return autograd::layer_norm(x, w, b, wrapped_norm_shape); };
        
        TestMetrics m = benchmark_op("LayerNorm", run_fn, x, expected.to(device), 100, 0.2f);
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 3;
        print_result("LayerNorm", m);
    }
    
    // --- Embedding ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        int64_t vocab = 1000;
        int64_t dim = 768;
        int64_t seq = 128;
        
        Shape w_shape{{vocab, dim}};
        Shape idx_shape{{B, seq}}; // or just flat
        
        Tensor w = Tensor::randn<float>(w_shape, opts);
        
        // Indices (CPU usually ok, but for fused we want GPU indices if supported)
        std::vector<int32_t> idx_data(B*seq);
        for(int i=0; i<B*seq; i++) idx_data[i] = i % vocab;
        
        Tensor indices = Tensor(idx_shape, opts.with_dtype(Dtype::Int32).with_req_grad(false));
        indices.set_data(idx_data);
         
        Tensor w_cpu = w.to_cpu();
        Tensor idx_cpu = indices.to_cpu();
        Tensor expected = autograd::embedding(w_cpu, idx_cpu);
        
        auto run_fn = [&]( ) { return autograd::embedding(w, indices); };
        
        TestMetrics m = benchmark_op("Embedding", run_fn, w, expected.to(device));
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 2;
        print_result("Embedding", m);
    }
}
