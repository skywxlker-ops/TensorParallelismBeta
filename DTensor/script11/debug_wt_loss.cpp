#include <iostream>
#include <cmath>
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/EmbeddingOps.h"

using namespace OwnTensor;

int main() {
    std::cout << "=== Weight Tying + Loss Test ===" << std::endl;
    
    DeviceIndex device(Device::CUDA, 0);
    cudaSetDevice(0);
    
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                        .with_device(device)
                                        .with_req_grad(true);
    
    int64_t B = 8;
    int64_t T = 1024;
    int64_t vocab_size = 50304;
    int64_t n_embd = 768;
    
    // Token embedding (like wte.weight in GPT)
    std::cout << "Creating wte.weight [" << vocab_size << ", " << n_embd << "]..." << std::endl;
    Tensor wte_weight = Tensor::randn<float>(Shape{{vocab_size, n_embd}}, opts, 1234, 0.02f);
    
    // Mock hidden states (like x after MLP blocks) [B, T, n_embd]
    std::cout << "Creating hidden states [" << B << ", " << T << ", " << n_embd << "]..." << std::endl;
    Tensor x = Tensor::randn<float>(Shape{{B, T, n_embd}}, opts, 5678, 0.1f);
    
    // Check x for NaN
    {
        Tensor cpu = x.to_cpu();
        float* d = cpu.data<float>();
        for (size_t i = 0; i < std::min<size_t>(1000, cpu.numel()); ++i) {
            if (std::isnan(d[i]) || std::isinf(d[i])) {
                std::cout << "NaN in x at " << i << std::endl;
                return 1;
            }
        }
        std::cout << "x OK (sample: " << d[0] << ")" << std::endl;
    }
    
    // Transpose wte for weight tying
    std::cout << "Transposing wte.weight..." << std::endl;
    Tensor wte_t = autograd::transpose(wte_weight, 0, 1);  // [n_embd, vocab_size]
    std::cout << "wte_t shape: [" << wte_t.shape().dims[0] << ", " << wte_t.shape().dims[1] << "]" << std::endl;
    
    // Check wte_t for NaN
    {
        Tensor cpu = wte_t.to_cpu();
        float* d = cpu.data<float>();
        for (size_t i = 0; i < std::min<size_t>(1000, cpu.numel()); ++i) {
            if (std::isnan(d[i]) || std::isinf(d[i])) {
                std::cout << "NaN in wte_t at " << i << std::endl;
                return 1;
            }
        }
        std::cout << "wte_t OK (sample: " << d[0] << ")" << std::endl;
    }
    
    // Matmul: x @ wte_t -> logits [B, T, vocab_size]
    std::cout << "Computing logits = x @ wte_t..." << std::endl;
    Tensor logits = autograd::matmul(x, wte_t);
    cudaDeviceSynchronize();
    
    std::cout << "logits shape: [" << logits.shape().dims[0] << ", " << logits.shape().dims[1] << ", " << logits.shape().dims[2] << "]" << std::endl;
    std::cout << "logits is_contiguous: " << logits.is_contiguous() << std::endl;
    
    // Check logits for NaN
    {
        Tensor cpu = logits.to_cpu();
        float* d = cpu.data<float>();
        bool found_nan = false;
        size_t nan_count = 0;
        for (size_t i = 0; i < cpu.numel(); ++i) {
            if (std::isnan(d[i]) || std::isinf(d[i])) {
                if (!found_nan) {
                    std::cout << "First NaN/Inf in logits at index " << i << ": " << d[i] << std::endl;
                    found_nan = true;
                }
                nan_count++;
            }
        }
        if (nan_count > 0) {
            std::cout << "Total NaN/Inf in logits: " << nan_count << " out of " << cpu.numel() << std::endl;
            return 1;
        }
        std::cout << "logits OK (sample: " << d[0] << ", " << d[1] << ", " << d[2] << ")" << std::endl;
    }
    
    // Create targets
    std::cout << "Creating targets..." << std::endl;
    Tensor targets = Tensor(Shape{{B, T}}, TensorOptions().with_dtype(Dtype::Int64).with_device(device));
    {
        Tensor targets_cpu(Shape{{B, T}}, TensorOptions().with_dtype(Dtype::Int64));
        int64_t* tdata = targets_cpu.data<int64_t>();
        for (int64_t i = 0; i < B * T; ++i) {
            tdata[i] = i % vocab_size;
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
    
    // Now test backward
    std::cout << "\nTesting backward..." << std::endl;
    Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device), 1.0f);
    loss.backward(&grad_scale);
    cudaDeviceSynchronize();
    
    // Check gradients
    if (wte_weight.has_grad()) {
        Tensor grad_cpu = wte_weight.grad_view().to_cpu();
        float* gdata = grad_cpu.data<float>();
        bool found_nan = false;
        size_t nan_count = 0;
        for (size_t i = 0; i < grad_cpu.numel(); ++i) {
            if (std::isnan(gdata[i]) || std::isinf(gdata[i])) {
                if (!found_nan) {
                    std::cout << "First NaN/Inf in wte grad at index " << i << ": " << gdata[i] << std::endl;
                    found_nan = true;
                }
                nan_count++;
            }
        }
        if (nan_count > 0) {
            std::cout << "Total NaN/Inf in wte grad: " << nan_count << " out of " << grad_cpu.numel() << std::endl;
        } else {
            std::cout << "wte_weight gradient OK (sample: " << gdata[0] << ")" << std::endl;
        }
    } else {
        std::cout << "WARNING: wte_weight has no gradient!" << std::endl;
    }
    
    if (x.has_grad()) {
        Tensor grad_cpu = x.grad_view().to_cpu();
        float* gdata = grad_cpu.data<float>();
        bool found_nan = false;
        for (size_t i = 0; i < std::min<size_t>(1000, grad_cpu.numel()); ++i) {
            if (std::isnan(gdata[i]) || std::isinf(gdata[i])) {
                std::cout << "NaN/Inf in x grad at index " << i << std::endl;
                found_nan = true;
                break;
            }
        }
        if (!found_nan) {
            std::cout << "x gradient OK (sample: " << gdata[0] << ")" << std::endl;
        }
    } else {
        std::cout << "WARNING: x has no gradient!" << std::endl;
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
