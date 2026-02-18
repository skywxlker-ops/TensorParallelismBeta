#include "autograd_test_utils.h"

void run_loss_tests();

void run_loss_tests() {
    std::cout << "\n=== Loss Function Tests ===\n" << std::endl;
    
    int64_t B = 64, C = 10;
    Shape shape{{B, C}};
    
    bool use_cuda = true;
#ifndef WITH_CUDA
    use_cuda = false;
#endif
    DeviceIndex device = use_cuda ? DeviceIndex(Device::CUDA, 0) : DeviceIndex(Device::CPU);
    TensorOptions opts = TensorOptions().with_device(device).with_dtype(Dtype::Float32).with_req_grad(true);

    // --- MSE ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        Tensor pred = Tensor::randn<float>(shape, opts);
        Tensor target = Tensor::randn<float>(shape, opts.with_req_grad(false));
         
        Tensor pred_cpu = pred.to_cpu();
        Tensor target_cpu = target.to_cpu();
        Tensor expected = autograd::mse_loss(pred_cpu, target_cpu);
        
        auto run_fn = [&]( ) { return autograd::mse_loss(pred, target); };
        
        TestMetrics m = benchmark_op("MSE", run_fn, pred, expected.to(device));
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 2;
        print_result("MSE", m);
    }
    
    // --- MAE ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        Tensor pred = Tensor::randn<float>(shape, opts);
        Tensor target = Tensor::randn<float>(shape, opts.with_req_grad(false));
        
        Tensor pred_cpu = pred.to_cpu();
        Tensor target_cpu = target.to_cpu();
        Tensor expected = autograd::mae_loss(pred_cpu, target_cpu);
        
        auto run_fn = [&]( ) { return autograd::mae_loss(pred, target); };
        
        TestMetrics m = benchmark_op("MAE", run_fn, pred, expected.to(device));
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 2;
        print_result("MAE", m);
    }
    
    // --- BCE ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        // Probabilities 0..1
        Tensor pred_raw = autograd::sigmoid(Tensor::randn<float>(shape, opts));
        Tensor pred = pred_raw.detach();
        pred.set_requires_grad(true);
        
        Tensor target = autograd::sigmoid(Tensor::randn<float>(shape, opts.with_req_grad(false))); // targets also 0..1
        
        Tensor pred_cpu = pred.to_cpu();
        Tensor target_cpu = target.to_cpu();
        Tensor expected = autograd::binary_cross_entropy(pred_cpu, target_cpu);
        
        auto run_fn = [&]( ) { return autograd::binary_cross_entropy(pred, target); };
        
        TestMetrics m = benchmark_op("BCE", run_fn, pred, expected.to(device));
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 2;
        print_result("BCE", m);
    }
    
    // --- CCE ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        // Softmax output
        Tensor pred_raw = autograd::softmax(Tensor::randn<float>(shape, opts), -1);
        Tensor pred = pred_raw.detach();
        pred.set_requires_grad(true);
        
        Tensor target = autograd::softmax(Tensor::randn<float>(shape, opts.with_req_grad(false)), -1); // One-hot or prob distribution
        
        Tensor pred_cpu = pred.to_cpu();
        Tensor target_cpu = target.to_cpu();
        Tensor expected = autograd::categorical_cross_entropy(pred_cpu, target_cpu);
        
        auto run_fn = [&]( ) { return autograd::categorical_cross_entropy(pred, target); };
        
        TestMetrics m = benchmark_op("CCE", run_fn, pred, expected.to(device));
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 2;
        print_result("CCE", m);
    }
    
    // --- Sparse CCE (GPT-2 style) ---
    {
        int64_t start_mem = Tensor::get_active_tensor_count();
        int64_t vocab = 1000;
        Shape logits_shape{{B, vocab}};
        Shape target_shape{{B}};
        
        Tensor logits = Tensor::randn<float>(logits_shape, opts);
        // Targets: random ints 0..vocab-1 (using float for now or construct specially)
        // We'll trust the randomness to be valid or careful construction
        // Need targets as Int64 or Int32.
        std::vector<int64_t> t_data(B);
        for(int i=0; i<B; i++) t_data[i] = i % vocab;
        Tensor targets = Tensor(target_shape, opts.with_dtype(Dtype::Int64).with_req_grad(false));
        targets.set_data(t_data);
         
        // Reference CPU
        Tensor logits_cpu = logits.to_cpu();
        Tensor targets_cpu = targets.to_cpu();
        Tensor expected = autograd::sparse_cross_entropy_loss(logits_cpu, targets_cpu);
        
        auto run_fn = [&]( ) { return autograd::sparse_cross_entropy_loss(logits, targets); };
        
        // Don't verify numerics strictly here as CPU fallback for sparse might vary slightly or is complex setup
        TestMetrics m = benchmark_op("Sparse CCE", run_fn, logits, expected.to(device));
        m.memory_leak_count = Tensor::get_active_tensor_count() - start_mem - 2;
        print_result("Sparse CCE", m);
    }
}
