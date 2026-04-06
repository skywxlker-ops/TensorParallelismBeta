#pragma once

#include "ops/helpers/MultiTensorKernels.h"
#include <vector>

namespace OwnTensor {
namespace cuda {

// Fused AdamW update with loss-scaler unscaling and gradient clipping folded in.
//
// Each gradient element is multiplied by (inv_scale * clip_coeff) before the
// Adam moment update and parameter step. This eliminates the separate per-param
// cast+scale kernel launches that are otherwise issued before multi_tensor_adam_cuda.
//
// Parameters:
//   params / grads / ms / vs   -- same layout as multi_tensor_adam_cuda
//   lr, beta1, beta2, eps, weight_decay, bias_correction1/2  -- Adam hyperparams
//   inv_scale   -- 1.0 / loss_scaler_scale  (pass 1.0 when no scaler)
//   clip_coeff  -- gradient clipping coefficient (pass 1.0 when no clipping)
//
// With inv_scale=1.0 and clip_coeff=1.0 the output is numerically identical to
// multi_tensor_adam_cuda(is_adamw=true).
void fused_adamw_with_unscale_cuda(
    const std::vector<TensorInfo>& params,
    const std::vector<TensorInfo>& grads,
    const std::vector<TensorInfo>& ms,
    const std::vector<TensorInfo>& vs,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float inv_scale,
    float clip_coeff
);

} // namespace cuda
} // namespace OwnTensor
