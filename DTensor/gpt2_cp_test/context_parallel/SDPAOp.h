#pragma once

#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/BinaryOps.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/TensorOps.h"
#include <cmath>
#include <limits>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// SDPAResult
//
// Holds the output of a single SDPA step:
//   out: attention output tensor  [B, H, T_q, D]
//   lse: log-sum-exp per query row [B, H, T_q, 1]  (used by SDPAMerger)
// ---------------------------------------------------------------------------
struct SDPAResult {
    Tensor out;  // [B, H, T_q, D]
    Tensor lse;  // [B, H, T_q, 1]
};


// ---------------------------------------------------------------------------
// sdpa_forward
//
// Scaled Dot-Product Attention for a single (q, k, v) chunk.
//
//   scores  = matmul(q * scale, k^T)        -- [B, H, T_q, T_k]
//   lse     = logsumexp(scores, dim=-1)      -- [B, H, T_q, 1]
//   masked  = tril_softmax(scores) or softmax(scores)
//   out     = matmul(masked, v)              -- [B, H, T_q, D]
//
// LSE computation: When is_causal=true, scores are masked with tril before
// computing LSE so that future K positions (upper triangle) do not inflate
// the normalization constant. This ensures the merger weights correctly
// represent the actual attention distribution for each ring step.
// ---------------------------------------------------------------------------
inline SDPAResult sdpa_forward(
    Tensor& q,
    Tensor& k,
    Tensor& v,
    bool is_causal,
    float scale)
{
    // Step 1: Compute attention scores: Q * K^T
    Tensor k_t = autograd::transpose(k, -2, -1);

    // Scale tensor (small constant, created once per call)
    Shape scale_shape({{1}});
    TensorOptions scale_opts = TensorOptions()
        .with_dtype(q.dtype())
        .with_device(q.device());
    Tensor scale_tensor = Tensor::full(scale_shape, scale_opts, scale);

    Tensor q_scaled = autograd::mul(q, scale_tensor);
    Tensor scores = autograd::matmul(q_scaled, k_t);

    // Step 2: Compute LSE (not autograd-tracked)
    // For causal: apply tril to mask future positions to -1e9 BEFORE
    // computing LSE. Without this, future positions inflate the LSE for
    // early-sequence tokens (by up to log(T_k) ~ 6.2 for position 0),
    // causing wrong merger weights and gradual training instability.
    // The tril here matches what fused_tril_softmax does internally for
    // the softmax probs -- we need it separately because the LSE is
    // computed on a different code path (non-autograd raw ops).
    int64_t last_dim = scores.ndim() - 1;
    Tensor scores_for_lse = is_causal
        ? OwnTensor::tril(scores, 0, -1e9)
        : scores;
    Tensor max_scores = reduce_max(scores_for_lse, {last_dim}, true);
    Tensor exp_shifted = OwnTensor::exp(scores_for_lse - max_scores);
    Tensor sum_exp = reduce_sum(exp_shifted, {last_dim}, true);
    Tensor lse = max_scores + OwnTensor::log(sum_exp);

    // Step 3: Apply causal mask + softmax (autograd-tracked)
    Tensor attn_probs;
    if (is_causal) {
        attn_probs = autograd::fused_tril_softmax(scores, 0);
    } else {
        attn_probs = autograd::softmax(scores, -1);
    }

    // Step 4: Compute output: attn_probs @ V
    Tensor out = autograd::matmul(attn_probs, v);

    return SDPAResult{out, lse};
}


// ---------------------------------------------------------------------------
// sdpa_backward_op
//
// Backward SDPA for a single (q, k, v) chunk during the backward ring loop.
// Recomputes the forward graph, then calls autograd::backward to produce
// grad_q, grad_k, grad_v.
//
// Returns: {grad_q, grad_k, grad_v}
// ---------------------------------------------------------------------------
inline std::vector<Tensor> sdpa_backward_op(
    Tensor& q,
    Tensor& k,
    Tensor& v,
    Tensor& grad_output,
    bool is_causal,
    float scale)
{
    // Recompute forward with autograd tracking
    Tensor k_t = autograd::transpose(k, -2, -1);

    Shape scale_shape({{1}});
    TensorOptions scale_opts = TensorOptions()
        .with_dtype(q.dtype())
        .with_device(q.device());
    Tensor scale_tensor = Tensor::full(scale_shape, scale_opts, scale);

    Tensor q_scaled = autograd::mul(q, scale_tensor);
    Tensor scores = autograd::matmul(q_scaled, k_t);

    Tensor attn_probs;
    if (is_causal) {
        attn_probs = autograd::fused_tril_softmax(scores, 0);
    } else {
        attn_probs = autograd::softmax(scores, -1);
    }

    Tensor out = autograd::matmul(attn_probs, v);

    // Run backward through the autograd graph
    autograd::backward(out, &grad_output);

    // Extract gradients
    Tensor grad_q = q.has_grad() ? q.grad_view() : Tensor();
    Tensor grad_k = k.has_grad() ? k.grad_view() : Tensor();
    Tensor grad_v = v.has_grad() ? v.grad_view() : Tensor();

    return {grad_q, grad_k, grad_v};
}