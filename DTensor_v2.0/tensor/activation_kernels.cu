#include "tensor/activation_kernels.h"
#include <cuda_runtime.h>
#include <cmath>

/**
 * GELU (Gaussian Error Linear Unit) activation kernel
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 */
__global__ void gelu_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        float x_cubed = x * x * x;
        float inner = 0.797885f * (x + 0.044715f * x_cubed);
        data[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/**
 * ReLU (Rectified Linear Unit) activation kernel
 */
__global__ void relu_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

/**
 * Fused Bias + GELU kernel (saves one memory pass)
 */
__global__ void bias_gelu_kernel(float* output, const float* bias,
                                  size_t output_size, size_t bias_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float x = output[idx] + bias[idx % bias_size];
        float x_cubed = x * x * x;
        float inner = 0.797885f * (x + 0.044715f * x_cubed);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/**
 * Fused Bias + ReLU kernel
 */
__global__ void bias_relu_kernel(float* output, const float* bias,
                                  size_t output_size, size_t bias_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        output[idx] = fmaxf(0.0f, output[idx] + bias[idx % bias_size]);
    }
}

/**
 * Bias-only kernel (no activation)
 */
__global__ void bias_kernel(float* output, const float* bias,
                            size_t output_size, size_t bias_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        output[idx] += bias[idx % bias_size];
    }
}

/**
 * Launch the appropriate activation kernel
 */
void launch_activation_kernel(float* data, size_t size, 
                              ActivationType activation, 
                              cudaStream_t stream) {
    if (activation == ActivationType::NONE) return;
    
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    
    switch (activation) {
        case ActivationType::GELU:
            gelu_kernel<<<num_blocks, threads_per_block, 0, stream>>>(data, size);
            break;
        case ActivationType::RELU:
            relu_kernel<<<num_blocks, threads_per_block, 0, stream>>>(data, size);
            break;
        default:
            break;
    }
}

/**
 * Launch fused bias+activation kernel
 */
void launch_bias_activation_kernel(float* output, const float* bias,
                                   size_t output_size, size_t bias_size,
                                   ActivationType activation,
                                   cudaStream_t stream) {
    const int threads_per_block = 256;
    const int num_blocks = (output_size + threads_per_block - 1) / threads_per_block;
    
    switch (activation) {
        case ActivationType::GELU:
            bias_gelu_kernel<<<num_blocks, threads_per_block, 0, stream>>>
                (output, bias, output_size, bias_size);
            break;
        case ActivationType::RELU:
            bias_relu_kernel<<<num_blocks, threads_per_block, 0, stream>>>
                (output, bias, output_size, bias_size);
            break;
        case ActivationType::NONE:
            bias_kernel<<<num_blocks, threads_per_block, 0, stream>>>
                (output, bias, output_size, bias_size);
            break;
    }
}

/**
 * Launch bias-only kernel
 */
void launch_bias_kernel(float* output, const float* bias,
                        size_t output_size, size_t bias_size,
                        cudaStream_t stream) {
    const int threads_per_block = 256;
    const int num_blocks = (output_size + threads_per_block - 1) / threads_per_block;
    
    bias_kernel<<<num_blocks, threads_per_block, 0, stream>>>
        (output, bias, output_size, bias_size);
}
