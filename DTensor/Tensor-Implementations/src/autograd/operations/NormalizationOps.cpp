#include "autograd/operations/NormalizationOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/NormalizationBackward.h"
#include "autograd/AutogradContext.h"
#include "autograd/Variable.h" // For make_variable
#include "ops/helpers/LayerNormKernels.h"
#include "dtype/DtypeTraits.h" // For is_same_type in TensorDataManip.h
#include "dtype/Types.h"
#include "core/TensorDataManip.h" // For data access

namespace OwnTensor {
namespace autograd {

Tensor layer_norm(
    const Tensor& input, 
    const Tensor& weight, 
    const Tensor& bias, 
    int normalized_shape, 
    float eps)
{
    // 1. Prepare Output Tensors
    Shape x_shape = input.shape();
    Tensor output = Tensor(x_shape, input.opts());
    
    // Mean and Rstd are (N,)
    // Assuming normalized_shape corresponds to the last dimension size.
    // Total rows N = numel / normalized_shape
    int64_t total_ele = input.numel();
    int64_t cols = normalized_shape;
    int64_t rows = total_ele / cols;
    
    // Check shape validity
    if (x_shape.dims.back() != cols) {
        throw std::runtime_error("LayerNorm: Last dimension of input must match normalized_shape");
    }
    
    Tensor mean = Tensor(Shape{{rows}}, input.opts().with_req_grad(false));
    Tensor rstd = Tensor(Shape{{rows}}, input.opts().with_req_grad(false));
    
    // 2. Dispatch
    if (input.device().is_cuda()) {
        const float* gamma_ptr = (weight.is_valid()) ? weight.data<float>() : nullptr;
        const float* beta_ptr = (bias.is_valid()) ? bias.data<float>() : nullptr;
        
        cuda::layer_norm_forward_cuda(
            input.data<float>(),
            gamma_ptr,
            beta_ptr,
            output.data<float>(),
            mean.data<float>(),
            rstd.data<float>(),
            rows,
            cols,
            eps
        );
    } else {
        // CUDA Bridge: Connection to run on CUDA if available
#ifdef WITH_CUDA
        try {
            // Move to CUDA (Device 0)
            DeviceIndex gpu_dev(Device::CUDA, 0);
            Tensor x_cu = input.to(gpu_dev);
            Tensor w_cu = (weight.is_valid()) ? weight.to(gpu_dev) : Tensor();
            Tensor b_cu = (bias.is_valid()) ? bias.to(gpu_dev) : Tensor();
            
            // Execute on GPU
            Tensor out_cu = layer_norm(x_cu, w_cu, b_cu, normalized_shape, eps);
            
            // Move back to original device (CPU)
            return out_cu.to(input.device());
        } catch (...) {
            // Fallback to CPU execution if CUDA fails
        }
#endif
        // CPU Fallback (OpenMP) 

        const float* x_ptr = input.data<float>();
        const float* gamma_ptr = (weight.is_valid()) ? weight.data<float>() : nullptr;
        const float* beta_ptr = (bias.is_valid()) ? bias.data<float>() : nullptr;
        float* y_ptr = output.data<float>();
        float* mean_ptr = mean.data<float>();
        float* rstd_ptr = rstd.data<float>();
        
        #pragma omp parallel for
        for (int64_t i = 0; i < rows; ++i) {
            const float* row_x = x_ptr + i * cols;
            float* row_y = y_ptr + i * cols;
            
            // Mean
            float sum = 0.0f;
            for (int64_t j = 0; j < cols; ++j) sum += row_x[j];
            float mu = sum / cols;
            mean_ptr[i] = mu;
            
            // Var
            float sum_sq = 0.0f;
            for (int64_t j = 0; j < cols; ++j) {
                float diff = row_x[j] - mu;
                sum_sq += diff * diff;
            }
            float var = sum_sq / cols;
            float rs = 1.0f / std::sqrt(var + eps);
            rstd_ptr[i] = rs;
            
            // Normalize
            for (int64_t j = 0; j < cols; ++j) {
                float val = (row_x[j] - mu) * rs;
                float g = (gamma_ptr) ? gamma_ptr[j] : 1.0f;
                float b = (beta_ptr) ? beta_ptr[j] : 0.0f;
                row_y[j] = val * g + b;
            }
        }
    }
    
// Construct Autograd Graph
    if (input.requires_grad() || (weight.is_valid() && weight.requires_grad()) || (bias.is_valid() && bias.requires_grad())) {
        
        auto grad_fn = std::make_shared<LayerNormBackward>(
            input, mean, rstd, weight, normalized_shape, eps
        );
        
        Tensor& input_mut = const_cast<Tensor&>(input);
        if (input.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(input_mut));
        }
        
        if (weight.is_valid() && weight.requires_grad()) {
            Tensor& weight_mut = const_cast<Tensor&>(weight);
            grad_fn->set_next_edge(1, get_grad_edge(weight_mut));
        }
        
        if (bias.is_valid() && bias.requires_grad()) {
            Tensor& bias_mut = const_cast<Tensor&>(bias);
            grad_fn->set_next_edge(2, get_grad_edge(bias_mut));
        }
        
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    
    return output;
}

} // namespace autograd
} // namespace OwnTensor