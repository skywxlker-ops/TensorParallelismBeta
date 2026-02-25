#include "nn/NN.h"
#include "autograd/AutogradOps.h"
#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"
#include <cmath>

namespace OwnTensor {
namespace nn {

// ============================================================================
// LayerNorm
// ============================================================================

LayerNorm::LayerNorm(int normalized_shape, float eps) : eps(eps) {
    TensorOptions opts = TensorOptions().with_req_grad(true);
    
    // Initialize weight (gamma) to ones and bias (beta) to zeros
    weight = Tensor::ones(Shape{{normalized_shape}}, opts);
    bias = Tensor::zeros(Shape{{normalized_shape}}, opts);
    
    register_parameter(weight);
    register_parameter(bias);
}

Tensor LayerNorm::forward(const Tensor& input) {
    // Use Fused LayerNorm Op (CUDA Optimized)
    int normalized_shape = weight.shape().dims[0];
    return autograd::layer_norm(input, weight, bias, normalized_shape, eps);
}

// parameters() and to() are handled by base Module since parameters are registered.

} // namespace nn
} // namespace OwnTensor
