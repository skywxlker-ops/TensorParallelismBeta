#include "autograd/operations/LossOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/LossBackward.h"
#include "ops/TensorOps.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/ScalarOps.h"

#ifdef WITH_CUDA
#include "ops/helpers/LossKernels.h"
#endif

#include "utils/Profiler.h"
#include "checkpointing/GradMode.h"
#include <iostream>

namespace OwnTensor {
namespace autograd {

Tensor mse_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.device().is_cuda() && predictions.dtype() == Dtype::Float32) {
         Tensor result = Tensor::zeros(Shape{{1}}, TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()));
         {
             AUTO_PROFILE_CUDA("Forward::MSE_Loss_CUDA");
             cuda::mse_loss_forward_cuda(predictions.data<float>(), targets.data<float>(), result.data<float>(), predictions.numel());
         }
         
         if (GradMode::is_enabled() && predictions.requires_grad()) {
             auto grad_fn = std::make_shared<MSELossBackward>(predictions, targets, predictions.numel());
             Tensor& pred_mut = const_cast<Tensor&>(predictions);
             grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
             result.set_grad_fn(grad_fn);
             result.set_requires_grad(true);
         }
         return result;
    }

    // Forward: mean((pred - target)^2)
    Tensor diff = predictions - targets;
    Tensor sq_diff = OwnTensor::pow(diff, 2, 0);
    Tensor result = reduce_mean(sq_diff);
    
    // Build graph if predictions require grad
    if (predictions.requires_grad()) {
        auto grad_fn = std::make_shared<MSELossBackward>(predictions, targets, predictions.numel());
        Tensor& pred_mut = const_cast<Tensor&>(predictions);
        grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor mae_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.device().is_cuda() && predictions.dtype() == Dtype::Float32) {
         Tensor result = Tensor::zeros(Shape{{1}}, TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()));
         cuda::mae_loss_forward_cuda(predictions.data<float>(), targets.data<float>(), result.data<float>(), predictions.numel());
         
         if (predictions.requires_grad()) {
             auto grad_fn = std::make_shared<MAELossBackward>(predictions, targets, predictions.numel());
             Tensor& pred_mut = const_cast<Tensor&>(predictions);
             grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
             result.set_grad_fn(grad_fn);
             result.set_requires_grad(true);
         }
         return result;
    }

    // Forward: mean(|pred - target|)
    Tensor diff = predictions - targets;
    Tensor abs_diff = OwnTensor::abs(diff, 0);
    Tensor result = reduce_mean(abs_diff);
    
    // Build graph if predictions require grad
    if (GradMode::is_enabled() && predictions.requires_grad()) {
        auto grad_fn = std::make_shared<MAELossBackward>(predictions, targets, predictions.numel());
        Tensor& pred_mut = const_cast<Tensor&>(predictions);
        grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets) {
    if (predictions.device().is_cuda() && predictions.dtype() == Dtype::Float32) {
         Tensor result = Tensor::zeros(Shape{{1}}, TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()));
         cuda::bce_loss_forward_cuda(predictions.data<float>(), targets.data<float>(), result.data<float>(), predictions.numel());
         
         if (GradMode::is_enabled() && predictions.requires_grad()) {
             auto grad_fn = std::make_shared<BCELossBackward>(predictions, targets, predictions.numel());
             Tensor& pred_mut = const_cast<Tensor&>(predictions);
             grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
             result.set_grad_fn(grad_fn);
             result.set_requires_grad(true);
         }
         return result;
    }

    float epsilon_val = 1e-7f;
    Tensor epsilon = Tensor::full(predictions.shape(), 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), epsilon_val);
    Tensor one_minus_epsilon = Tensor::full(predictions.shape(), 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), 1.0f - epsilon_val);

    // Clip predictions
    Tensor clipped_preds = where((predictions < epsilon), epsilon, predictions);
    clipped_preds = where((clipped_preds > one_minus_epsilon), one_minus_epsilon, clipped_preds);

    Tensor term1 = targets * OwnTensor::log(clipped_preds);
    Tensor ones = Tensor::ones(targets.shape(), 
        TensorOptions().with_device(targets.device()).with_dtype(targets.dtype()));
    Tensor term2 = (ones - targets) * OwnTensor::log(ones - clipped_preds);
    Tensor sum_terms = term1 + term2;
    Tensor neg_one = Tensor::full({{1}}, 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), -1.0f);
    Tensor result = reduce_mean(sum_terms) * neg_one;
    
    // Build graph if predictions require grad
    if (predictions.requires_grad()) {
        auto grad_fn = std::make_shared<BCELossBackward>(clipped_preds, targets, predictions.numel());
        Tensor& pred_mut = const_cast<Tensor&>(predictions);
        grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor categorical_cross_entropy(const Tensor& predictions, const Tensor& targets) {
    if (predictions.device().is_cuda() && predictions.dtype() == Dtype::Float32) {
         Tensor result = Tensor::zeros(Shape{{1}}, TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()));
         
         int64_t num_classes = predictions.shape().dims.back();
         int64_t batch_size = predictions.numel() / num_classes;
         
         cuda::categorical_cross_entropy_forward_cuda(
             predictions.data<float>(),
             targets.data<float>(),
             result.data<float>(),
             batch_size, num_classes
         );
         
         // Build graph if predictions require grad
         if (predictions.requires_grad()) {
             auto grad_fn = std::make_shared<CCELossBackward>(predictions, targets, predictions.numel());
             Tensor& pred_mut = const_cast<Tensor&>(predictions);
             grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
             result.set_grad_fn(grad_fn);
             result.set_requires_grad(true);
         }
         return result;
    }

    float epsilon_val = 1e-7f;
    Tensor epsilon = Tensor::full(predictions.shape(), 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), epsilon_val);
    Tensor one_minus_epsilon = Tensor::full(predictions.shape(), 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), 1.0f - epsilon_val);

    // Clip predictions
    Tensor clipped_preds = where((predictions < epsilon), epsilon, predictions);
    clipped_preds = where((clipped_preds > one_minus_epsilon), one_minus_epsilon, clipped_preds);

    Tensor log_preds = OwnTensor::log(clipped_preds);
    Tensor target_log_probs = targets * log_preds;
    
    std::vector<int64_t> axis = {1};
    Tensor sample_losses = reduce_sum(target_log_probs, axis);
    Tensor neg_one = Tensor::full({{1}}, 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), -1.0f);
    Tensor result = reduce_mean(sample_losses) * neg_one;
    
    // Build graph if predictions require grad
    if (GradMode::is_enabled() && predictions.requires_grad()) {
        auto grad_fn = std::make_shared<CCELossBackward>(clipped_preds, targets, predictions.numel());
        Tensor& pred_mut = const_cast<Tensor&>(predictions);
        grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor sparse_cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    // Handle both 2D [N, C] and 3D [B, T, C] logits
    auto logits_shape = logits.shape().dims;
    int64_t batch_size, num_classes;
    Tensor logits_2d = logits;
    Tensor targets_1d = targets;
    
    if (logits_shape.size() == 3) {
        // 3D logits [B, T, C] -> flatten to [B*T, C]
        int64_t B = logits_shape[0];
        int64_t T = logits_shape[1];
        num_classes = logits_shape[2];
        batch_size = B * T;
        logits_2d = logits.view(Shape{{batch_size, num_classes}});
        
        // Flatten targets [B, T] -> [B*T] if needed
        auto targets_shape = targets.shape().dims;
        if (targets_shape.size() == 2) {
            targets_1d = targets.view(Shape{{batch_size}});
        }
    } else if (logits_shape.size() == 2) {
        batch_size = logits_shape[0];
        num_classes = logits_shape[1];
    } else {
        throw std::runtime_error("sparse_cross_entropy_loss: logits must be 2D [N, C] or 3D [B, T, C]");
    }
    
    auto targets_shape = targets_1d.shape().dims;
    if (targets_shape.size() != 1 || targets_shape[0] != batch_size) {
        throw std::runtime_error("sparse_cross_entropy_loss: targets shape mismatch");
    }
    
    // Compute softmax and cross entropy loss
    // For numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    TensorOptions opts = TensorOptions()
        .with_dtype(logits.dtype())
        .with_device(logits.device());
    
    float total_loss = 0.0f;
    
    if (logits_2d.device().is_cpu()) {
        const float* logits_data = logits_2d.data<float>();
        
        for (int64_t i = 0; i < batch_size; ++i) {
            // Find max for numerical stability
            float max_val = logits_data[i * num_classes];
            for (int64_t c = 1; c < num_classes; ++c) {
                max_val = std::max(max_val, logits_data[i * num_classes + c]);
            }
            
            // Compute log-sum-exp
            float sum_exp = 0.0f;
            for (int64_t c = 0; c < num_classes; ++c) {
                sum_exp += std::exp(logits_data[i * num_classes + c] - max_val);
            }
            float log_sum_exp = max_val + std::log(sum_exp);
            
            // Get target class
            int64_t target_class = 0;
            if (targets_1d.dtype() == Dtype::Int64) {
                target_class = targets_1d.data<int64_t>()[i];
            } else if (targets_1d.dtype() == Dtype::Int32) {
                target_class = static_cast<int64_t>(targets_1d.data<int32_t>()[i]);
            } else if (targets_1d.dtype() == Dtype::UInt16) {
                target_class = static_cast<int64_t>(targets_1d.data<uint16_t>()[i]);
            }
            
            // Loss = log_sum_exp - logits[target]
            float loss_i = log_sum_exp - logits_data[i * num_classes + target_class];
            total_loss += loss_i;
        }
    } else {
        // CUDA: Use optimized CUDA kernel
        #ifdef WITH_CUDA
        // Keep result on GPU to avoid sync
        Tensor result = Tensor::zeros(Shape{{1}}, opts);
        
        if (logits_2d.dtype() == Dtype::Float32) {
            cudaStream_t stream = 0; // default for now
            if (targets_1d.dtype() == Dtype::UInt16) {
                cuda::sparse_cross_entropy_forward_cuda<float, uint16_t>(
                    logits_2d.data<float>(), targets_1d.data<uint16_t>(),
                    result.data<float>(), batch_size, num_classes, stream);
            } else if (targets_1d.dtype() == Dtype::Int64) {
                cuda::sparse_cross_entropy_forward_cuda<float, int64_t>(
                    logits_2d.data<float>(), targets_1d.data<int64_t>(),
                    result.data<float>(), batch_size, num_classes, stream);
            } else if (targets_1d.dtype() == Dtype::Int32) {
                cuda::sparse_cross_entropy_forward_cuda<float, int32_t>(
                    logits_2d.data<float>(), targets_1d.data<int32_t>(),
                    result.data<float>(), batch_size, num_classes, stream);
            } else {
                throw std::runtime_error("sparse_cross_entropy_loss: unsupported target dtype for CUDA");
            }
        } else {
            throw std::runtime_error("sparse_cross_entropy_loss: only Float32 supported for CUDA forward pass");
        }
        
        // Division by batch_size on GPU (simple scalar op)
        result = result / static_cast<float>(batch_size);

        // Build autograd graph
        if (GradMode::is_enabled() && logits.requires_grad()) {
            auto grad_fn = std::make_shared<SparseCrossEntropyLossBackward>(
                logits, targets_1d, batch_size, num_classes);
            Tensor& logits_mut = const_cast<Tensor&>(logits);
            grad_fn->set_next_edge(0, get_grad_edge(logits_mut));
            result.set_grad_fn(grad_fn);
            result.set_requires_grad(true);
        }
        return result;
        #else
        throw std::runtime_error("CUDA not available but tensor is on CUDA device");
        #endif
    }
    
    // Average loss
    float mean_loss = total_loss / static_cast<float>(batch_size);
    
    // Create scalar result tensor
    Tensor result = Tensor::full(Shape{{1}}, opts, mean_loss);
    
    // Build autograd graph
    if (GradMode::is_enabled() && logits.requires_grad()) {
        auto grad_fn = std::make_shared<SparseCrossEntropyLossBackward>(
            logits, targets_1d, batch_size, num_classes);
        Tensor& logits_mut = const_cast<Tensor&>(logits);
        grad_fn->set_next_edge(0, get_grad_edge(logits_mut));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

} // namespace autograd
} // namespace OwnTensor