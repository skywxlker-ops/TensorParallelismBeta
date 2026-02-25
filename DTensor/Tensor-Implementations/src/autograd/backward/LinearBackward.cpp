#include "autograd/backward/LinearBackward.h"
#include "utils/Profiler.h"
#include "ops/TensorOps.h"
#include "ops/Kernels.h"
#include "ops/UnaryOps/Reduction.h"
#ifdef WITH_CUDA
#include "ops/MatmulBackward.cuh"
#include "ops/LinearKernels.cuh"
#endif
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

LinearBackward::LinearBackward(const Tensor& input, const Tensor& weight)
    : Node(3),  // 3 inputs potentially: input, weight, bias (though bias doesn't need saving for backward)
      saved_input_(input),
      saved_weight_(weight) {}

std::vector<Tensor> LinearBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("LinearBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    Tensor grad_input;
    Tensor grad_weight;
    bool computed_main_grads = false;
    
#ifdef WITH_CUDA
    // Optimized CUDA path for Linear layer patterns
    if (grad_output.is_cuda() && saved_weight_.ndim() == 2) {
        // Case 1: Pure 2D matmul
        if (saved_input_.ndim() == 2 && grad_output.ndim() == 2) {
            grad_input = Tensor(saved_input_.shape(), saved_input_.dtype(), saved_input_.device());
            grad_weight = Tensor(saved_weight_.shape(), saved_weight_.dtype(), saved_weight_.device());
            cuda_matmul_backward(grad_output, saved_input_, saved_weight_, grad_input, grad_weight, 0);
            computed_main_grads = true;
        }
        // Case 2: Linear layer [B,T,Hidden]
        else if (saved_input_.ndim() > 2 && grad_output.ndim() == saved_input_.ndim()) {
            int64_t hidden_dim = saved_input_.shape().dims.back();
            int64_t output_dim = grad_output.shape().dims.back();
            
            Tensor a_flat = saved_input_.reshape(Shape{{-1, hidden_dim}});
            Tensor g_flat = grad_output.reshape(Shape{{-1, output_dim}});
            
            Tensor grad_input_flat(a_flat.shape(), a_flat.dtype(), a_flat.device());
            grad_weight = Tensor(saved_weight_.shape(), saved_weight_.dtype(), saved_weight_.device());
            
            cuda_matmul_backward(g_flat, a_flat, saved_weight_, grad_input_flat, grad_weight, 0);
            
            grad_input = grad_input_flat.reshape(saved_input_.shape());
            computed_main_grads = true;
        }
    }
#endif

    std::vector<Tensor> result;
    result.reserve(3);
    
    // 1. Gradient for Input
    if (next_edges_[0].is_valid()) {
        if (computed_main_grads) {
            result.push_back(grad_input);
        } else {
            AUTO_PROFILE_CUDA("Backward::Linear_GradInput_CUDA");
            result.push_back(matmul(grad_output, saved_weight_.t()));
        }
    } else {
        result.push_back(Tensor());
    }
    
    // 2. Gradient for Weight
    if (next_edges_[1].is_valid()) {
        if (computed_main_grads) {
            result.push_back(grad_weight);
        } else {
            AUTO_PROFILE_CUDA("Backward::Linear_GradWeight_CUDA");
            if (grad_output.ndim() == 3) {
                Tensor flat_input = saved_input_.reshape(Shape{{-1, saved_input_.shape().dims.back()}});
                Tensor flat_grad = grad_output.reshape(Shape{{-1, grad_output.shape().dims.back()}});
                result.push_back(matmul(flat_input.t(), flat_grad));
            } else {
                result.push_back(matmul(saved_input_.t(), grad_output));
            }
        }
    } else {
        result.push_back(Tensor());
    }
    
    // 3. Gradient for Bias
    if (next_edges_.size() > 2 && next_edges_[2].is_valid()) {
#ifdef WITH_CUDA
        if (grad_output.is_cuda()) {
             AUTO_PROFILE_CUDA("Backward::Linear_GradBias_CUDA");
             // Create output tensor
             Tensor grad_bias = Tensor(Shape{{grad_output.shape().dims.back()}}, grad_output.dtype(), grad_output.device());
             cuda_linear_bias_backward(grad_output, grad_bias, 0); // stream 0
             result.push_back(grad_bias);
        } else 
#endif
        {
            std::vector<int64_t> dims_to_sum;
            for (int i = 0; i < grad_output.ndim() - 1; ++i) {
                dims_to_sum.push_back(i);
            }
            result.push_back(reduce_sum(grad_output, dims_to_sum, false));
        }
    } else {
        result.push_back(Tensor());
    }
    
    return result;
}

} // namespace autograd
} // namespace OwnTensor
