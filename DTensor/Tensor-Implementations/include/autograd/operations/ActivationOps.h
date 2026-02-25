#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware ReLU
 */
Tensor relu(const Tensor& x);

/**
 * @brief Autograd-aware GeLU
 */
Tensor gelu(const Tensor& x);

/**
 * @brief Autograd-aware sigmoid
 */
Tensor sigmoid(const Tensor& x);

/**
 * @brief Autograd-aware softmax
 */
Tensor softmax(const Tensor& x, int64_t dim = -1);

} // namespace autograd
} // namespace OwnTensor
