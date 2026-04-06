#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backend selection for scaled_dot_product_attention.
 *
 * Math:             Composes existing ops (matmul + fused_tril_softmax).
 *                   Materializes the full T×T attention matrix.
 * MemoryEfficient:  Custom CUDA kernel using online softmax.
 *                   Avoids T×T allocation — O(T) extra memory.
 */
enum class SDPBackend {
    Math,
    MemoryEfficient,
    // FlashAttention,  // Future
};

/**
 * @brief Autograd-aware scaled dot-product attention.
 *
 * Computes: softmax(Q @ K^T / sqrt(head_dim)) @ V
 * with optional causal mask.
 *
 * @param query     (B, nh, T, hd) query tensor
 * @param key       (B, nh, T, hd) key tensor (S instead of T for cross-attention)
 * @param value     (B, nh, T, hd) value tensor
 * @param is_causal If true, applies lower-triangular causal mask
 * @param backend   Which implementation to use
 * @return          (B, nh, T, hd) attention output
 */
Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal = true,
    float dropout_p = 0.0f,
    SDPBackend backend = SDPBackend::Math
);

} // namespace autograd
} // namespace OwnTensor