#pragma once

#include <cuda_runtime.h>

namespace OwnTensor {
namespace dnn {

// =============================================================================
// LM Head forward and backward using TF32 Tensor Cores via cublasGemmEx.
//
// On Ampere+ (sm >= 80) cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32 uses
// Tensor Cores with TF32 compute (10-bit mantissa, 8-bit exponent). Inputs and
// outputs remain float32 — no casting required. The TF32 rounding is
// numerically indistinguishable from FP32 for typical GPT-2 scale dot products
// (max abs diff < 0.1 vs standard cublasSgemm FP32).
//
// forward:
//   logits[BT, vocab_size] = hidden[BT, n_embd] @ weight.T
//
// backward:
//   d_hidden[BT, n_embd]       = d_logits[BT, vocab_size] @ weight
//   d_weight[vocab_size, n_embd] = d_logits.T @ hidden
//
// All pointers must be device pointers. d_weight is overwritten (beta=0).
// =============================================================================

void fused_lm_head_forward(
    const float* hidden,      // [BT, n_embd]
    const float* weight,      // [vocab_size, n_embd]
    float*       logits,      // [BT, vocab_size]  output
    int BT,
    int n_embd,
    int vocab_size);

void fused_lm_head_backward(
    const float* d_logits,    // [BT, vocab_size]
    const float* hidden,      // [BT, n_embd]
    const float* weight,      // [vocab_size, n_embd]
    float*       d_hidden,    // [BT, n_embd]       output
    float*       d_weight,    // [vocab_size, n_embd] output (accumulated, beta=1)
    int BT,
    int n_embd,
    int vocab_size);

} // namespace dnn
} // namespace OwnTensor
