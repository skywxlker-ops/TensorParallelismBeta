#include "autograd/backward/NormalizationBackward.h"
#include "autograd/operations/NormalizationOps.h"
#include "ops/helpers/LayerNormKernels.h"
#include "dtype/DtypeTraits.h"
#include "dtype/Types.h"
#include "core/TensorDataManip.h"

namespace OwnTensor {
namespace autograd {

std::vector<Tensor> LayerNormBackward::apply(std::vector<Tensor>&& grads) {
    // 1. Unpack
    Tensor grad_output = grads[0];
    Tensor input = input_.unpack(shared_from_this()); // Version check
    Tensor mean = mean_.unpack(shared_from_this());
    Tensor rstd = rstd_.unpack(shared_from_this());
    
    Tensor weight;
    if (weight_.defined()) weight = weight_.unpack(shared_from_this());
    
    int64_t total_ele = input.numel();
    int64_t cols = normalized_shape_;
    int64_t rows = total_ele / cols;
    
    // 2. Output Gradients
    Tensor grad_input = Tensor::zeros(input.shape(), input.opts());
    Tensor grad_weight, grad_bias;
    
    if (weight.is_valid()) {
        grad_weight = Tensor::zeros(weight.shape(), weight.opts());
        grad_bias = Tensor::zeros(weight.shape(), weight.opts());
    } else {
        // Even if no weight provided, we might need grad_bias if it was provided?
        // In this architecture, we usually assume if weight exists, we compute it.
        // If bias was separate... logic becomes complex.
        // For LayerNorm, weight and bias usually go together.
        // If the forward didn't have weight, we treat gamma=1.
        // But we need to return grads matching the forward connectivity.
        // Forward inputs: input, weight, bias.
        // We need 3 grads.
        grad_weight = Tensor::zeros(Shape{{cols}}, input.opts());
        grad_bias = Tensor::zeros(Shape{{cols}}, input.opts());
    }
    
    // 3. Dispatch
    if (grad_output.device().is_cuda()) {
        const float* gamma_ptr = (weight.is_valid()) ? weight.data<float>() : nullptr;
        float* grad_gamma_ptr = (weight.is_valid()) ? grad_weight.data<float>() : nullptr;
        float* grad_beta_ptr = (weight.is_valid()) ? grad_bias.data<float>() : nullptr;
        
        // If weight wasn't valid in forward but bias was, we have a mismatch in logic.
        // Current implementation assumes weight/bias are paired or properly handled.
        // Simplification: if we need to return grad_bias, we must compute it.
        // The Kernel handles nullptr pointers safely.
        
        // But we need to verify if we *should* compute them?
        // We compute everything requested. The Autograd engine will discard what's not needed (via edges).
        // Standard practice: compute all connected gradients.
        
        if (!weight.is_valid()) {
             // If no weight param, we should still compute grad_bias if bias existed?
             // Not storing bias in SavedVariable makes this tricky.
             // We'll compute into temporary buffer if needed or just allocated zeros.
        }
        
        cuda::layer_norm_backward_cuda(
            grad_output.data<float>(),
            input.data<float>(),
            mean.data<float>(),
            rstd.data<float>(),
            gamma_ptr,
            grad_input.data<float>(),
            grad_weight.data<float>(),
            grad_bias.data<float>(),
            rows,
            cols
        );
        
    } else {
        // TODO: CPU fallback
        // CPU Fallback
        const float* gy_ptr = grad_output.data<float>();
        const float* x_ptr = input.data<float>();
        const float* mean_ptr = mean.data<float>();
        const float* rstd_ptr = rstd.data<float>();
        const float* gamma_ptr = (weight.is_valid()) ? weight.data<float>() : nullptr;
        
        float* gx_ptr = grad_input.data<float>();
        float* gw_ptr = grad_weight.data<float>();
        float* gb_ptr = grad_bias.data<float>();
        
        // Zero init grads
        // (Assuming Tensor::zeros already zeroes memory)
        
        // Parallel over rows for gx and accumulation for gw/gb
        // Naive CPU implementation:
        // 1. Compute gw/gb
        std::vector<float> gw_acc(cols, 0.0f);
        std::vector<float> gb_acc(cols, 0.0f);
        
        // This is slow on CPU (atomic or reduction needed). 
        // Single thread for reduction for simplicity on CPU (not performance critical path here).
        for (int64_t i = 0; i < rows; ++i) {
             const float* row_gy = gy_ptr + i * cols;
             const float* row_x = x_ptr + i * cols;
             float mu = mean_ptr[i];
             float rs = rstd_ptr[i];
             
             for(int64_t j=0; j<cols; ++j) {
                 float val = (row_x[j] - mu) * rs;
                 float gy = row_gy[j];
                 
                 gw_acc[j] += gy * val;
                 gb_acc[j] += gy;
             }
        }
        
        // Copy to output
        for(int64_t j=0; j<cols; ++j) {
            gw_ptr[j] = gw_acc[j];
            gb_ptr[j] = gb_acc[j];
        }

        // 2. Compute gx
        #pragma omp parallel for
        for (int64_t i = 0; i < rows; ++i) {
             const float* row_gy = gy_ptr + i * cols;
             const float* row_x = x_ptr + i * cols;
             float* row_gx = gx_ptr + i * cols;
             float mu = mean_ptr[i];
             float rs = rstd_ptr[i];
             
             float sum1 = 0.0f; // sum(gy * gamma)
             float sum2 = 0.0f; // sum(gy * gamma * x_hat)
             
             for(int64_t j=0; j<cols; ++j) {
                 float g = (gamma_ptr) ? gamma_ptr[j] : 1.0f;
                 float gy = row_gy[j];
                 float val = (row_x[j] - mu) * rs;
                 
                 sum1 += gy * g;
                 sum2 += gy * g * val;
             }
             
             for(int64_t j=0; j<cols; ++j) {
                 float g = (gamma_ptr) ? gamma_ptr[j] : 1.0f;
                 float gy = row_gy[j];
                 float val = (row_x[j] - mu) * rs;
                 
                 row_gx[j] = rs * (gy * g - (sum1 + val * sum2) / cols);
             }
        }
    }
    
    return {grad_input, grad_weight, grad_bias};
}

} // namespace autograd
} // namespace OwnTensor
