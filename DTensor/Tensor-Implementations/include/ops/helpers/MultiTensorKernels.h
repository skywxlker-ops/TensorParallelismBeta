#pragma once
#include <vector>
#include <cstdint>

namespace OwnTensor {
namespace cuda {

struct TensorInfo {
   float* ptr;
   int64_t numel;
};

/**
* @brief Compute total L2 norm squared of multiple tensors
*/
void multi_tensor_grad_norm_cuda(
   const std::vector<TensorInfo>& tensors,
   float* norm_sq_accumulator
);

/**
* @brief Scale multiple tensors by a single coefficient stored on GPU
*/
void multi_tensor_scale_cuda(
   const std::vector<TensorInfo>& tensors,
   const float* clip_coef
);

/**
* @brief Fused Adam update for multiple tensors
*/
void multi_tensor_adam_cuda(
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
   bool is_adamw = false
);

} // namespace cuda
} // namespace OwnTensor
