#pragma once

// ---------------------------------------------------------------------------
// FusedSDPAOp.h
//
// C++ wrapper that calls the FlashAttention CUDA kernel and returns the same
// SDPAResult{out, lse} struct as sdpa_forward() in SDPAOp.h.
//
// Drop-in replacement for sdpa_forward() in the hot path.  The returned
// tensors are NOT connected to an autograd graph (the kernel bypasses it).
// For the backward pass continue using sdpa_backward_op_manual() from
// SDPAOp.h -- it recomputes the unfused forward internally.
//
// Interface:
//
//   SDPAResult sdpa_fused_forward(
//       Tensor& q,          // [B, H, T_q, D]
//       Tensor& k,          // [B, H, T_k, D]
//       Tensor& v,          // [B, H, T_k, D]
//       bool    is_causal,
//       float   scale,
//       int     q_offset = 0,   // global sequence start of Q chunk (CP use)
//       int     k_offset = 0);  // global sequence start of K chunk (CP use)
//
// q_offset / k_offset are used only for causal masking during ring-attention
// CP steps where Q and K/V may come from different positions in the sequence.
// For standard (non-CP) use leave them at 0.
//
// LSE shape: [B, H, T_q, 1]  -- matches SDPAMerger::step() expectations.
// ---------------------------------------------------------------------------

#include "core/Tensor.h"
#include "gpt2_cp_test/context_parallel/SDPAOp.h"
#include "gpt2_cp_test/context_parallel/FusedSDPAKernel.h"
#include "gpt2_cp_test/context_parallel/FusedSDPABackwardKernel.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// sdpa_fused_forward
// ---------------------------------------------------------------------------
inline SDPAResult sdpa_fused_forward(
    Tensor& q,
    Tensor& k,
    Tensor& v,
    bool    is_causal,
    float   scale,
    int     q_offset = 0,
    int     k_offset = 0)
{
    // --- input validation ---------------------------------------------------
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
        throw std::runtime_error(
            "sdpa_fused_forward: Q, K, V must be 4-D tensors [B, H, T, D]");
    }
    if (q.dtype() != Dtype::Float32) {
        throw std::runtime_error(
            "sdpa_fused_forward: only Float32 is supported");
    }

    const int64_t B   = q.shape().dims[0];
    const int64_t H   = q.shape().dims[1];
    const int64_t T_q = q.shape().dims[2];
    const int64_t D   = q.shape().dims[3];
    const int64_t T_k = k.shape().dims[2];

    if (k.shape().dims[0] != B || k.shape().dims[1] != H || k.shape().dims[3] != D ||
        v.shape().dims[0] != B || v.shape().dims[1] != H ||
        v.shape().dims[2] != T_k || v.shape().dims[3] != D) {
        throw std::runtime_error(
            "sdpa_fused_forward: K/V shape mismatch with Q");
    }
    if (D > 256) {
        throw std::runtime_error(
            "sdpa_fused_forward: head dim > 256 not supported");
    }

    const int BH = static_cast<int>(B * H);

    // --- allocate output tensors on the same device as Q -------------------
    TensorOptions base_opts = q.opts().with_req_grad(false);

    Shape out_shape({{B, H, T_q, D}});
    Tensor out = Tensor::empty(out_shape, base_opts);

    // LSE shape: [B, H, T_q, 1]  (keepdim convention used by SDPAMerger)
    Shape lse_shape({{B, H, T_q, 1}});
    Tensor lse = Tensor::empty(lse_shape, base_opts);

    // --- get raw float* pointers (device memory) ---------------------------
    // The kernel treats the layout as [BH, T, D] by multiplying the B and H
    // dimensions into the first axis -- valid because Tensor storage is
    // row-major and contiguous.
    const float* Q_ptr = q.data<float>();
    const float* K_ptr = k.data<float>();
    const float* V_ptr = v.data<float>();
    float*       O_ptr = out.data<float>();

    // LSE kernel output is flat [BH, T_q]; we write into the lse tensor's
    // storage directly (its total element count equals BH * T_q because the
    // trailing 1-dim is just a keepdim view).
    float* LSE_ptr = lse.data<float>();

    // --- launch kernel ------------------------------------------------------
    launch_flash_attn_fwd_f32(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, LSE_ptr,
        BH,
        static_cast<int>(T_q),
        static_cast<int>(T_k),
        static_cast<int>(D),
        scale,
        is_causal,
        q_offset,
        k_offset);

    // Synchronise so the caller can safely read results immediately
    cudaDeviceSynchronize();

    return SDPAResult{out, lse};
}

// ---------------------------------------------------------------------------
// sdpa_fused_backward
//
// Fused FlashAttention backward.  Replaces sdpa_backward_op_manual() for the
// hot path in ContextParallelBackward.
//
// Inputs:
//   q, k, v        -- saved forward inputs      [B, H, T_q/k, D]
//   grad_out       -- incoming gradient dO       [B, H, T_q, D]
//   out            -- merged forward output O    [B, H, T_q, D]
//   merged_lse     -- merged log-sum-exp         [B, H, T_q, 1]
//
// For CP ring steps, P_ij = exp(s_ij - merged_lse_i) -- the merger
// rescaling (exp(step_lse - merged_lse)) is baked into the LSE directly,
// so no separate lse_diff tensor is needed.
//
// Supported D: 32, 64, 128.
// ---------------------------------------------------------------------------
inline std::vector<Tensor> sdpa_fused_backward(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& grad_out,
    const Tensor& out,
    const Tensor& merged_lse,
    bool    is_causal,
    float   scale,
    int     q_offset = 0,
    int     k_offset = 0)
{
    const int64_t B   = q.shape().dims[0];
    const int64_t H   = q.shape().dims[1];
    const int64_t T_q = q.shape().dims[2];
    const int64_t D   = q.shape().dims[3];
    const int64_t T_k = k.shape().dims[2];
    const int BH = static_cast<int>(B * H);

    if (D > 256) {
        throw std::runtime_error(
            "sdpa_fused_backward: head dim > 256 not supported");
    }

    TensorOptions base_opts = q.opts().with_req_grad(false);

    Tensor dQ = Tensor::zeros(q.shape(),               base_opts);
    Tensor dK = Tensor::zeros(k.shape(),               base_opts);
    Tensor dV = Tensor::zeros(v.shape(),               base_opts);

    // D_buf [BH, T_q] -- scratch written by dQ kernel, read by dK/dV kernel
    Shape d_shape({{static_cast<int64_t>(BH), T_q}});
    Tensor D_buf = Tensor::empty(d_shape, base_opts);

    launch_flash_attn_bwd_f32(
        q.data<float>(), k.data<float>(), v.data<float>(),
        out.data<float>(), grad_out.data<float>(), merged_lse.data<float>(),
        dQ.data<float>(), dK.data<float>(), dV.data<float>(),
        D_buf.data<float>(),
        BH,
        static_cast<int>(T_q),
        static_cast<int>(T_k),
        static_cast<int>(D),
        scale, is_causal, q_offset, k_offset);

    cudaDeviceSynchronize();

    return {dQ, dK, dV};
}