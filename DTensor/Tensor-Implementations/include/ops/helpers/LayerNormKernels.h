#pragma once
#include <cuda_runtime.h>

namespace OwnTensor {
namespace cuda {

// Forward LayerNorm
// x: [rows, cols]
// gamma: [cols] (can be nullptr)
// beta: [cols] (can be nullptr)
// y: [rows, cols] (output)
// mean: [rows] (output)
// rstd: [rows] (output)
void layer_norm_forward_cuda(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    float* mean,
    float* rstd,
    int rows,
    int cols,
    float eps);

// Backward LayerNorm
// grad_y: [rows, cols]
// x: [rows, cols]
// mean: [rows]
// rstd: [rows]
// gamma: [cols] (can be nullptr)
// grad_x: [rows, cols] (output)
// grad_gamma: [cols] (output, can be nullptr)
// grad_beta: [cols] (output, can be nullptr)
void layer_norm_backward_cuda(
    const float* grad_y,
    const float* x,
    const float* mean,
    const float* rstd,
    const float* gamma,
    float* grad_x,
    float* grad_gamma,
    float* grad_beta,
    int rows,
    int cols);

} // namespace cuda
} // namespace OwnTensor
