#pragma once
#include "core/Tensor.h"

namespace OwnTensor {
namespace dnn {

// Autograd-integrated layer norm using vectorized kernels (VectorizedLayerNormKernel.cu).
// Drop-in for autograd::layer_norm / nn::LayerNorm::forward.
//
// forward:  y = (x - mean) * rstd * gamma + beta    [float32 only]
// backward: uses vln_backward_f32 (warp-shuffle, float4 vectorized)
//
// gamma (weight) and beta (bias) must not be nullptr.
// cols must equal x.shape().dims.back().
Tensor fused_layer_norm(
    const Tensor& x,
    const Tensor& gamma,
    const Tensor& beta,
    int cols,
    float eps = 1e-5f);

} // namespace dnn
} // namespace OwnTensor
