#include "autograd/backward/ActivationBackward.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/UnaryOps/Trigonometry.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/helpers/ActivationKernels.h"
#include <stdexcept>
#include <cmath>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// ReluBackward
// ============================================================================

ReluBackward::ReluBackward(const Tensor& input)
    : Node(1), saved_input_(input) {}

std::vector<Tensor> ReluBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("ReluBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_input = grad_output * (input > 0)
    // grad_input = grad_output * (input > 0)
    Tensor grad_input;
    if (grad_output.device().is_cuda() && grad_output.dtype() == Dtype::Float32) {
         grad_input = Tensor(saved_input_.shape(), grad_output.opts());
         cuda::relu_backward_cuda(grad_output.data<float>(), saved_input_.data<float>(), grad_input.data<float>(), grad_input.numel());
    } else {
         Tensor mask = saved_input_ > 0.0f;
         grad_input = grad_output * mask;
    }
    
    return {grad_input};
}

// ============================================================================
// GeLUBackward
// ============================================================================

GeLUBackward::GeLUBackward(const Tensor& input)
    : Node(1), saved_input_(input) {}

std::vector<Tensor> GeLUBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("GeLUBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    const Tensor& x = saved_input_;
    
    // Use fused CUDA kernel for GPU tensors (much faster)
    if (x.device().is_cuda() && x.dtype() == Dtype::Float32) {
        Tensor grad_input(x.shape(), TensorOptions()
            .with_dtype(x.dtype())
            .with_device(x.device()));
        
        cuda::fused_gelu_backward_cuda(
            grad_output.data<float>(),
            x.data<float>(),
            grad_input.data<float>(),
            x.numel()
        );
        
        return {grad_input};
    }
    
    // TODO: CPU fallback
    // Fallback to tensor ops for CPU or non-float32
    // GeLU derivative:
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Let u = sqrt(2/pi) * (x + 0.044715 * x^3)
    // gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
    
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    const float c = 0.044715f;
    
    Tensor x_sq = x * x;
    Tensor x_cubed = x_sq * x;
    Tensor u = sqrt_2_over_pi * (x + c * x_cubed);
    Tensor tanh_u = tanh(u);
    
    // sech^2(u) = 1 - tanh^2(u)
    Tensor sech2_u = 1.0f - tanh_u * tanh_u;
    
    // du/dx = sqrt(2/pi) * (1 + 3*c*x^2)
    Tensor du_dx = sqrt_2_over_pi * (1.0f + 3.0f * c * x_sq);
    
    // gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
    Tensor grad_x = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;
    
    return {grad_output * grad_x};
}

// ============================================================================
// SigmoidBackward
// ============================================================================

SigmoidBackward::SigmoidBackward(const Tensor& output)
    : Node(1), saved_output_(output) {}

std::vector<Tensor> SigmoidBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SigmoidBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_x = grad_out * sigmoid(x) * (1 - sigmoid(x))
    // We saved sigmoid(x) as saved_output_
    Tensor grad_x;
    if (grad_output.device().is_cuda() && grad_output.dtype() == Dtype::Float32) {
         grad_x = Tensor(saved_output_.shape(), grad_output.opts());
         cuda::sigmoid_backward_cuda(grad_output.data<float>(), saved_output_.data<float>(), grad_x.data<float>(), grad_x.numel());
    } else {
         grad_x = grad_output * saved_output_ * (1.0f - saved_output_);
    }
    
    return {grad_x};
}

// ============================================================================
// SoftmaxBackward
// ============================================================================

SoftmaxBackward::SoftmaxBackward(const Tensor& output, int64_t dim)
    : Node(1), saved_output_(output), dim_(dim) {}

std::vector<Tensor> SoftmaxBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SoftmaxBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    const Tensor& s = saved_output_;  // softmax output
    
    // Softmax backward: grad_x = s * (grad_out - sum(grad_out * s, dim))
    Tensor grad_x;
    int64_t ndim = s.ndim();
    int64_t d = dim_ < 0 ? dim_ + ndim : dim_;
    
    if (grad_output.device().is_cuda() && grad_output.dtype() == Dtype::Float32 && d == ndim - 1) {
         grad_x = Tensor(s.shape(), grad_output.opts());
         int64_t cols = s.shape().dims.back();
         int64_t rows = s.numel() / cols;
         
         cuda::softmax_backward_cuda(grad_output.data<float>(), s.data<float>(), grad_x.data<float>(), rows, cols);
    } else {
         Tensor gs = grad_output * s;
         Tensor sum_gs = reduce_sum(gs, {dim_}, true);
         grad_x = s * (grad_output - sum_gs);
    }
    
    return {grad_x};
}

} // namespace autograd
} // namespace OwnTensor
