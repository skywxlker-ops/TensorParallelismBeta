#include <iostream>
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "autograd/operations/ReshapeOps.h"

using namespace OwnTensor;

int main() {
    std::cout << "=== Debug Loss Test ===" << std::endl;
    
    DeviceIndex device(Device::CUDA, 0);
    cudaSetDevice(0);
    
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                        .with_device(device)
                                        .with_req_grad(true);
    
    int64_t B = 2;   // Match GPT-2 validation
    int64_t T = 128; // Match GPT-2 validation
    int64_t vocab_size = 50304;
    
    // Create random logits [B, T, vocab_size]
    std::cout << "Creating logits..." << std::endl;
    Tensor logits = Tensor::randn<float>(Shape{{B, T, vocab_size}}, opts, 1234, 0.1f);
    
    // Check logits for NaN
    {
        Tensor cpu = logits.to_cpu();
        float* d = cpu.data<float>();
        bool has_nan = false;
        for (size_t i = 0; i < std::min<size_t>(1000, cpu.numel()); ++i) {
            if (std::isnan(d[i]) || std::isinf(d[i])) {
                std::cout << "NaN in logits at " << i << std::endl;
                has_nan = true;
                break;
            }
        }
        if (!has_nan) std::cout << "No NaN in logits sample ✓" << std::endl;
    }
    
    // Create random targets [B, T] with values in [0, vocab_size) using UInt16
    std::cout << "Creating targets (UInt16)..." << std::endl;
    Tensor targets = Tensor(Shape{{B, T}}, TensorOptions().with_dtype(Dtype::UInt16).with_device(device));
    {
        Tensor targets_cpu(Shape{{B, T}}, TensorOptions().with_dtype(Dtype::UInt16));
        uint16_t* tdata = targets_cpu.data<uint16_t>();
        for (int64_t i = 0; i < B * T; ++i) {
            tdata[i] = static_cast<uint16_t>(i % vocab_size);  // Valid indices 
        }
        targets = targets_cpu.to(device);
    }
    
    // Check targets
    {
        Tensor cpu = targets.to_cpu();
        uint16_t* d = cpu.data<uint16_t>();
        bool out_of_range = false;
        for (size_t i = 0; i < cpu.numel(); ++i) {
            if (d[i] >= vocab_size) { // UInt16 is always >= 0
                std::cout << "Out-of-range target at " << i << ": " << d[i] << std::endl;
                out_of_range = true;
                break;
            }
        }
        if (!out_of_range) std::cout << "All targets in range ✓" << std::endl;
    }
    
    // Compute loss
    std::cout << "Computing loss..." << std::endl;
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
    
    std::cout << "=== Test Passed ===" << std::endl;
    return 0;
}
