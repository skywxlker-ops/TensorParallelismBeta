#pragma once

#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "autograd/Variable.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <cmath>
#include <iomanip>

using namespace OwnTensor;

struct TestMetrics {
    double forward_latency_ms;
    double backward_latency_ms;
    double throughput_ops_sec; // For B*T size
    bool numerical_match;
    size_t memory_leak_count;
};

class Timer {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    TimePoint start_time;

public:
    void start() { start_time = Clock::now(); }
    double elapsed_ms() {
        auto end_time = Clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

inline bool verify_numerics(const Tensor& a, const Tensor& b, float tol = 1e-4) {
    if (a.numel() != b.numel()) {
        std::cerr << "Shape mismatch: " << a.numel() << " vs " << b.numel() << std::endl;
        return false;
    }
    
    Tensor a_cpu = a.to_cpu();
    Tensor b_cpu = b.to_cpu();
    
    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.numel(); ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            std::cerr << "Mismatch at index " << i << ": " << a_data[i] << " vs " << b_data[i] << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    // std::cout << "Max diff: " << max_diff << " ";
    return true;
}

inline TestMetrics benchmark_op(const std::string& name, 
                         std::function<Tensor()> forward_fn, 
                         Tensor& input_tensor,
                         const Tensor& expected_output,
                         int iterations = 100,
                         float tolerance = 1e-3) {
    (void)name;
    TestMetrics metrics = {0, 0, 0, true, 0};
    
    // Warmup
    for(int i=0; i<5; i++) {
        Tensor out = forward_fn();
        if (out.requires_grad()) {
             Tensor grad = Tensor::ones(out.shape(), out.opts());
             out.backward(&grad);
        }
        if (out.is_cuda()) cudaDeviceSynchronize();
    }
    
    // Check Numerics (assuming expected_output is provided/valid)
    if (expected_output.numel() > 0) {
        Tensor out = forward_fn();
        // Use loose tolerance for fast-math kernels like GELU/Sigmoid/Softmax which might use __expf etc
        metrics.numerical_match = verify_numerics(out, expected_output, tolerance);
    }

    // Time Forward
    Timer timer;
    if (input_tensor.is_cuda()) cudaDeviceSynchronize();
    
    timer.start();
    for(int i=0; i<iterations; i++) {
        Tensor out = forward_fn();
    }
    if (input_tensor.is_cuda()) cudaDeviceSynchronize();
    metrics.forward_latency_ms = timer.elapsed_ms() / iterations;
    
    // Time Backward (one iter for now due to graph overhead, or simpler loop)
    // Re-run forward to get fresh graph
    Tensor out_for_back = forward_fn();
    Tensor grad_out = Tensor::ones(out_for_back.shape(), out_for_back.opts());
    
    if (input_tensor.is_cuda()) cudaDeviceSynchronize();
    timer.start();
    for(int i=0; i<iterations; i++) {
        // We need to re-run forward to create graph node each time strictly? 
        // Or can we backward multiple times? PyTorch destroys graph.
        // So we must include forward in backward loop or re-forward.
        // To measure ONLY backward, we'd need to pre-construct graph but that's hard.
        // We'll verify backward latency by (Forward+Backward) - (Forward).
        Tensor loop_out = forward_fn();
        loop_out.backward(&grad_out); 
    }
    if (input_tensor.is_cuda()) cudaDeviceSynchronize();
    double total_ms = timer.elapsed_ms() / iterations;
    metrics.backward_latency_ms = total_ms - metrics.forward_latency_ms;
    if (metrics.backward_latency_ms < 0) metrics.backward_latency_ms = 0; // measurement noise

    // Throughput (Elements / sec)
    // Approx
    metrics.throughput_ops_sec = (input_tensor.numel() / (metrics.forward_latency_ms / 1000.0));

    // Memory Check
    // Should be checked outside by the caller comparing active counts
    
    return metrics;
}

inline void print_result(const std::string& name, const TestMetrics& m) {
    std::cout << std::left << std::setw(25) << name 
              << " | Fwd: " << std::fixed << std::setprecision(3) << m.forward_latency_ms << " ms"
              << " | Bwd: " << m.backward_latency_ms << " ms"
              << " | Thr: " << std::scientific << std::setprecision(2) << m.throughput_ops_sec << " elem/s"
              << " | Match: " << (m.numerical_match ? "PASS" : "FAIL")
              << " | Leaks: " << m.memory_leak_count
              << std::endl;
}
