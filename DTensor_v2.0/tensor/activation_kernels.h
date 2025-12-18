#pragma once
#include <cuda_runtime.h>

enum class ActivationType {
    GELU,
    RELU,
    NONE
};

/**
 * Launch activation kernel in-place on GPU tensor data
 */
void launch_activation_kernel(float* data, size_t size, 
                              ActivationType activation, 
                              cudaStream_t stream);

/**
 * Launch fused bias+activation kernel (saves one memory pass)
 * output[i] = activation(output[i] + bias[i % bias_size])
 */
void launch_bias_activation_kernel(float* output, const float* bias,
                                   size_t output_size, size_t bias_size,
                                   ActivationType activation,
                                   cudaStream_t stream);

/**
 * Launch bias-only kernel (no activation)
 * output[i] += bias[i % bias_size]
 */
void launch_bias_kernel(float* output, const float* bias,
                        size_t output_size, size_t bias_size,
                        cudaStream_t stream);
