#include <iostream>
#include <cmath>
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "autograd/operations/ReshapeOps.h"

using namespace OwnTensor;

int main() {
    std::cout << "=== Debug Loss with UInt16 Targets ===" << std::endl;
    
    DeviceIndex device(Device::CUDA, 0);
    cudaSetDevice(0);
    
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                        .with_device(device)
                                        .with_req_grad(true);
    
    int64_t B = 8;
    int64_t T = 1024;
    int64_t vocab_size = 50304;
    
    // Create logits [B, T, vocab_size] - similar scale to model output
    std::cout << "Creating logits..." << std::endl;
    Tensor logits = Tensor::randn<float>(Shape{{B, T, vocab_size}}, opts, 1234, 0.3f);
    
    // Check logits statistics
    {
        Tensor cpu = logits.to_cpu();
        float* d = cpu.data<float>();
        float min_v = d[0], max_v = d[0];
        for (size_t i = 0; i < cpu.numel(); ++i) {
            if (d[i] < min_v) min_v = d[i];
            if (d[i] > max_v) max_v = d[i];
        }
        std::cout << "Logits range: [" << min_v << ", " << max_v << "]" << std::endl;
    }
    
    // Create UInt16 targets (same dtype as data loader)
    std::cout << "Creating UInt16 targets..." << std::endl;
    Tensor targets(Shape{{B, T}}, TensorOptions().with_dtype(Dtype::UInt16).with_device(device));
    {
        Tensor targets_cpu(Shape{{B, T}}, TensorOptions().with_dtype(Dtype::UInt16));
        uint16_t* tdata = targets_cpu.data<uint16_t>();
        for (int64_t i = 0; i < B * T; ++i) {
            // Random targets in valid range
            tdata[i] = static_cast<uint16_t>(i % vocab_size);
        }
        targets = targets_cpu.to(device);
    }
    
    // Compute loss
    std::cout << "Computing sparse_cross_entropy_loss..." << std::endl;
    Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
    cudaDeviceSynchronize();
    
    // Check loss
    Tensor loss_cpu = loss.to_cpu();
    float loss_val = loss_cpu.data<float>()[0];
    std::cout << "Loss: " << loss_val << std::endl;
    
    if (std::isnan(loss_val) || std::isinf(loss_val)) {
        std::cout << "ERROR: Loss is NaN or Inf!" << std::endl;
        return 1;
    }
    
    // Test backward
    std::cout << "\nTesting backward..." << std::endl;
    loss.backward();
    cudaDeviceSynchronize();
    
    // Check gradients
    if (logits.has_grad()) {
        Tensor grad_cpu = logits.grad_view().to_cpu();
        float* gdata = grad_cpu.data<float>();
        bool found_nan = false;
        size_t nan_count = 0;
        for (size_t i = 0; i < grad_cpu.numel(); ++i) {
            if (std::isnan(gdata[i]) || std::isinf(gdata[i])) {
                if (!found_nan) {
                    std::cout << "First NaN/Inf in logits grad at index " << i << ": " << gdata[i] << std::endl;
                    found_nan = true;
                }
                nan_count++;
            }
        }
        if (nan_count > 0) {
            std::cout << "Total NaN/Inf in grad: " << nan_count << " out of " << grad_cpu.numel() << std::endl;
        } else {
            std::cout << "logits gradient OK (sample: " << gdata[0] << ")" << std::endl;
        }
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
