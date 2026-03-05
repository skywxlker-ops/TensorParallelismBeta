#pragma once
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Fused Layer Normalization
 * 
 * @param input Input tensor [..., normalized_shape]
 * @param weight Gamma parameter (optional, size [normalized_shape])
 * @param bias Beta parameter (optional, size [normalized_shape])
 * @param normalized_shape Size of the last dimension to normalize over
 * @param eps Epsilon for numerical stability
 * @return Tensor Normalized tensor
 */
Tensor layer_norm(
    const Tensor& input, 
    const Tensor& weight, 
    const Tensor& bias, 
    int normalized_shape, 
    float eps = 1e-5f);

} // namespace autograd
} // namespace OwnTensor
