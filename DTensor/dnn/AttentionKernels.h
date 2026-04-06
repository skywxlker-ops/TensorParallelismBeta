#pragma once
#include <cstdint>

namespace OwnTensor {
namespace cuda {

/**
 * @brief Compute log-sum-exp per row of attention scores.
 *
 * Read-only kernel — does NOT modify scores.
 * For causal: masks positions j > qi (where qi = row % T) as -inf.
 *
 * @param scores   (batch, T, T) pre-scaled attention scores
 * @param lse      (batch, T) output log-sum-exp per row
 * @param batch    number of (batch, head) slices (i.e., B * nh)
 * @param T        sequence length
 * @param is_causal  whether to apply causal mask
 */
void compute_row_lse(
    const float* scores,
    float* lse,
    int64_t batch,
    int64_t T,
    bool is_causal
);

/**
 * @brief Fused memory-efficient attention forward pass (FlashAttention-style).
 *
 * Computes: O = softmax(Q @ K^T / sqrt(hd)) @ V
 * WITHOUT materializing the full T×T attention matrix.
 *
 * Algorithm: Online softmax with tiled Q/K/V in shared memory.
 *   For each query block, iterates over key/value blocks:
 *     1. Q_block @ K_block^T  (tiled matmul in shared memory)
 *     2. Scale by 1/sqrt(hd)
 *     3. Apply causal mask
 *     4. Online softmax update (running max + sum)
 *     5. exp(scores - max) @ V_block accumulated into output
 *   Final normalization: O /= running_sum
 *
 * Optimizations:
 *   - No T×T allocation: O(T) extra memory instead of O(T²)
 *   - Shared memory tiling: Q loaded once, K/V reuse same buffer per iteration
 *   - exp2f(x * log2(e)): single-instruction fast exponentiation
 *   - Output accumulator in registers (no shared memory round-trip)
 *   - Warp shuffle reductions for row max/sum (4 threads per query row)
 *   - Causal early termination: skip key blocks beyond the diagonal
 *
 * @param query    (B, nh, T, hd) query tensor, contiguous
 * @param key      (B, nh, T, hd) key tensor, contiguous
 * @param value    (B, nh, T, hd) value tensor, contiguous
 * @param output   (B, nh, T, hd) output tensor (pre-allocated)
 * @param lse      (B, nh, T) log-sum-exp per row (pre-allocated, for backward)
 * @param B        batch size
 * @param nh       number of heads
 * @param T        sequence length
 * @param hd       head dimension (must be ≤ 256)
 * @param is_causal     whether to apply lower-triangular causal mask
 * @param dropout_p     dropout probability in [0, 1); pass 0 to disable
 * @param dropout_mask  pre-generated mask tensor of shape [B*nh, T, T] where
 *                      each element is either 0 (dropped) or 1/(1-dropout_p)
 *                      (kept and rescaled); pass nullptr when dropout_p == 0
 */
void mem_efficient_attn_forward(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* lse,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal,
   float dropout_p = 0.0f,
   const float* dropout_mask = nullptr
);

/**
 * @brief Fused memory-efficient attention forward pass — WMMA TF32 tensor-core variant.
 *
 * Identical inputs/outputs to mem_efficient_attn_forward but replaces the
 * scalar Q@K^T and P@V matrix multiplications with nvcuda::wmma mma_sync
 * using wmma::precision::tf32 (10-bit mantissa, FP32 accumulator).
 *
 * FP32 input data is converted to TF32 automatically by load_matrix_sync;
 * no explicit conversion required at the call site.
 *
 * Requirements:
 *   - Ampere GPU (sm_80+) — the build already targets sm_86.
 *   - HeadDim divisible by 16 (WMMA tile size). For other head dims this
 *     function falls back transparently to the scalar kernel.
 *   - HeadDim ≤ 256 (same constraint as the scalar kernel).
 *
 * Precision: TF32 accumulation introduces ~10^-3 relative error per
 * matrix multiply vs FP32.  Expect max absolute error < ~0.01 for
 * typical transformer inputs.
 *
 * @see mem_efficient_attn_forward for full parameter documentation.
 */
void mem_efficient_attn_forward_tc(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* lse,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal,
    float dropout_p = 0.0f,
    const float* dropout_mask = nullptr,
    int q_offset = 0,
    int k_offset = 0
);

/**
 * @brief Unified memory-efficient attention backward pass for dQ, dK, dV.
 *
 * Architecture (PyTorch CUTLASS FMHA style):
 *   1. Precompute D[i] = dot(dO[i], O[i]) in a lightweight kernel (one warp per row)
 *   2. Single KV-tile-centric kernel computes dQ, dK, dV together:
 *      - Each block owns one KV tile, sweeps all Q rows
 *      - Phase A: warp-parallel score recompute + dQ accumulation in registers
 *      - Phase B: cooperative dK/dV reduction (no per-element atomics)
 *      - dQ written via atomicAdd; dK/dV written once from shared memory
 *
 * @param D_buf    (B*nh, T) scratch buffer for precomputed D values (pre-allocated)
 */
void mem_efficient_attn_backward(
    const float* query,
    const float* key,
    const float* value,
    const float* output,
    const float* grad_output,
    const float* lse,
    float* grad_query,
    float* grad_key,
    float* grad_value,
    float* D_buf,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal,
    int q_offset = 0,
    int k_offset = 0
);

} // namespace cuda
} // namespace OwnTensor