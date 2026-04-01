// ---------------------------------------------------------------------------
// FusedSDPABackwardKernel.cu
//
// CUDA implementation of the FlashAttention backward pass.
//
// Avoids materialising the [T_q x T_k] attention matrix in HBM by tiling
// over K/V (for dQ) and Q (for dK/dV), recomputing P_ij on-the-fly from
// the saved merged LSE.
//
// Algorithm (Dao et al. 2022, FlashAttention backward):
//
//   D_i     = rowsum(dO_i * O_i)                    [precomputed in dQ kernel]
//   P_ij    = exp(s_ij - LSE_i)                      s_ij = dot(Q_i, K_j)*scale
//   dV_j   += P_ij * dO_i
//   ds_ij   = P_ij * (dot(dO_i, V_j) - D_i)
//   dQ_i   += ds_ij * scale * K_j
//   dK_j   += ds_ij * scale * Q_i
//
// Two kernels:
//   flash_attn_bwd_dq_kernel   -- one thread per Q row, sweeps K/V tiles.
//                                 Also writes D_buf[BH, T_q].
//   flash_attn_bwd_dkdv_kernel -- one thread per K row, sweeps Q tiles.
//                                 Reads D_buf.
//
// Both kernels launched in stream 0 (sequential).  No explicit sync between
// them -- same-stream ordering guarantees D_buf is ready before dK/dV reads.
//
// Template parameter HEAD_DIM is dispatched at launch for 32, 64, 128.
// ---------------------------------------------------------------------------

#include "FusedSDPABackwardKernel.h"
#include <cuda_runtime.h>
#include <math.h>

static constexpr int BLOCK_Q = 32;
static constexpr int BLOCK_K = 32;

// ---------------------------------------------------------------------------
// flash_attn_bwd_dq_kernel
//
// Thread threadIdx.x owns Q row q_local = blockIdx.x * BLOCK_Q + threadIdx.x.
// Sweeps all K/V tiles left-to-right (respecting causal mask with early exit).
// Writes dQ and D_buf.
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void flash_attn_bwd_dq_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ LSE,
    float* __restrict__       dQ,
    float* __restrict__       D_buf,
    int T_q, int T_k,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset)
{
    const int bh      = blockIdx.y;
    const int q_local = blockIdx.x * BLOCK_Q + threadIdx.x;

    if (q_local >= T_q) return;

    const int q_global = q_offset + q_local;

    // Shared memory: K_tile and V_tile
    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + BLOCK_K * HEAD_DIM;

    // Load Q, dO, O rows into registers
    const float* Q_ptr  = Q  + (bh * T_q + q_local) * HEAD_DIM;
    const float* dO_ptr = dO + (bh * T_q + q_local) * HEAD_DIM;
    const float* O_ptr  = O  + (bh * T_q + q_local) * HEAD_DIM;

    float q_regs[HEAD_DIM];
    float do_regs[HEAD_DIM];
    float dq_regs[HEAD_DIM];

#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        q_regs[d]  = Q_ptr[d];
        do_regs[d] = dO_ptr[d];
        dq_regs[d] = 0.0f;
    }

    // D_i = dot(dO_i, O_i) -- scalar, written to D_buf for dK/dV kernel
    float D_i = 0.0f;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) D_i += do_regs[d] * O_ptr[d];

    D_buf[bh * T_q + q_local] = D_i;

    const float lse_i = LSE[bh * T_q + q_local];

    // Sweep over K/V tiles
    const int num_k_blocks = (T_k + BLOCK_K - 1) / BLOCK_K;

    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_block_start  = kb * BLOCK_K;
        const int k_global_start = k_offset + k_block_start;

        // Causal early exit: entire tile is strictly in the future
        if (is_causal && k_global_start > q_global) break;

        // Cooperative tile load: thread t loads K[t] and V[t]
        const int k_local_t = k_block_start + threadIdx.x;
        if (k_local_t < T_k) {
            const float* Kp = K + (bh * T_k + k_local_t) * HEAD_DIM;
            const float* Vp = V + (bh * T_k + k_local_t) * HEAD_DIM;
            float* Ks = K_tile + threadIdx.x * HEAD_DIM;
            float* Vs = V_tile + threadIdx.x * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) { Ks[d] = Kp[d]; Vs[d] = Vp[d]; }
        } else {
            float* Ks = K_tile + threadIdx.x * HEAD_DIM;
            float* Vs = V_tile + threadIdx.x * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) { Ks[d] = 0.0f; Vs[d] = 0.0f; }
        }
        __syncthreads();

        const int tile_size = (k_block_start + BLOCK_K <= T_k)
                            ? BLOCK_K : (T_k - k_block_start);

        for (int j = 0; j < tile_size; ++j) {
            const int k_global_j = k_global_start + j;
            if (is_causal && k_global_j > q_global) continue;

            const float* Kj = K_tile + j * HEAD_DIM;
            const float* Vj = V_tile + j * HEAD_DIM;

            // s_ij = dot(Q_i, K_j) * scale
            float dot_qk = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dot_qk += q_regs[d] * Kj[d];
            const float p_ij = expf(dot_qk * scale - lse_i);

            // dp_ij = dot(dO_i, V_j)
            float dp_ij = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dp_ij += do_regs[d] * Vj[d];

            // ds_ij = p_ij * (dp_ij - D_i)
            const float ds_scaled = p_ij * (dp_ij - D_i) * scale;

#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dq_regs[d] += ds_scaled * Kj[d];
        }

        __syncthreads();
    }

    // Write dQ
    float* dQ_ptr = dQ + (bh * T_q + q_local) * HEAD_DIM;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) dQ_ptr[d] = dq_regs[d];
}

// ---------------------------------------------------------------------------
// flash_attn_bwd_dkdv_kernel
//
// Thread threadIdx.x owns K row k_local = blockIdx.x * BLOCK_K + threadIdx.x.
// Sweeps all Q tiles, accumulating dK_j and dV_j in registers.
// Reads D_buf (written by dQ kernel, guaranteed by same-stream ordering).
//
// Shared memory: Q_tile [BLOCK_Q*HEAD_DIM], dO_tile [BLOCK_Q*HEAD_DIM],
//                LSE_tile [BLOCK_Q], D_tile [BLOCK_Q].
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void flash_attn_bwd_dkdv_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D_buf,
    float* __restrict__       dK,
    float* __restrict__       dV,
    int T_q, int T_k,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset,
    const float* __restrict__ K)
{
    const int bh      = blockIdx.y;
    const int k_local = blockIdx.x * BLOCK_K + threadIdx.x;

    if (k_local >= T_k) return;

    const int k_global = k_offset + k_local;

    // Shared memory layout:
    //   [0               .. BLOCK_Q*HEAD_DIM) : Q_tile
    //   [BLOCK_Q*HEAD_DIM.. 2*BLOCK_Q*HD)     : dO_tile
    //   [2*BLOCK_Q*HD    .. 2*BLOCK_Q*HD+BQ)  : LSE_tile (floats)
    //   [2*BLOCK_Q*HD+BQ .. 2*BLOCK_Q*HD+2BQ) : D_tile  (floats)
    extern __shared__ float smem[];
    float* Q_tile   = smem;
    float* dO_tile  = smem + BLOCK_Q * HEAD_DIM;
    float* LSE_tile = smem + 2 * BLOCK_Q * HEAD_DIM;
    float* D_tile   = LSE_tile + BLOCK_Q;

    // Load K_j, V_j into registers (fixed across Q tiles)
    const float* K_ptr = K + (bh * T_k + k_local) * HEAD_DIM;
    const float* V_ptr = V + (bh * T_k + k_local) * HEAD_DIM;

    float k_regs[HEAD_DIM];
    float v_regs[HEAD_DIM];
    float dk_regs[HEAD_DIM];
    float dv_regs[HEAD_DIM];

#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        k_regs[d]  = K_ptr[d];
        v_regs[d]  = V_ptr[d];
        dk_regs[d] = 0.0f;
        dv_regs[d] = 0.0f;
    }

    const int num_q_blocks = (T_q + BLOCK_Q - 1) / BLOCK_Q;

    for (int qb = 0; qb < num_q_blocks; ++qb) {
        const int q_block_start  = qb * BLOCK_Q;
        const int q_global_start = q_offset + q_block_start;

        // Causal: skip Q tile if all Q rows are strictly before k_global
        // max q_global in tile = q_global_start + BLOCK_Q - 1
        if (is_causal && (q_global_start + BLOCK_Q - 1) < k_global) continue;

        // Cooperative tile load: thread t loads Q row and dO row at
        // q_block_start + threadIdx.x
        const int q_local_t = q_block_start + threadIdx.x;
        if (q_local_t < T_q) {
            const float* Qp  = Q  + (bh * T_q + q_local_t) * HEAD_DIM;
            const float* dOp = dO + (bh * T_q + q_local_t) * HEAD_DIM;
            float* Qt  = Q_tile  + threadIdx.x * HEAD_DIM;
            float* dOt = dO_tile + threadIdx.x * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) { Qt[d] = Qp[d]; dOt[d] = dOp[d]; }
            LSE_tile[threadIdx.x] = LSE[bh * T_q + q_local_t];
            D_tile[threadIdx.x]   = D_buf[bh * T_q + q_local_t];
        } else {
            float* Qt  = Q_tile  + threadIdx.x * HEAD_DIM;
            float* dOt = dO_tile + threadIdx.x * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) { Qt[d] = 0.0f; dOt[d] = 0.0f; }
            // Large LSE makes p_ij -> 0 for out-of-bounds rows
            LSE_tile[threadIdx.x] = 1e30f;
            D_tile[threadIdx.x]   = 0.0f;
        }
        __syncthreads();

        const int tile_size = (q_block_start + BLOCK_Q <= T_q)
                            ? BLOCK_Q : (T_q - q_block_start);

        for (int i = 0; i < tile_size; ++i) {
            const int q_global_i = q_global_start + i;
            // Causal: K position j attends Q position i only if k_global <= q_global_i
            if (is_causal && k_global > q_global_i) continue;

            const float* Qi  = Q_tile  + i * HEAD_DIM;
            const float* dOi = dO_tile + i * HEAD_DIM;
            const float  lse_i = LSE_tile[i];
            const float  D_i   = D_tile[i];

            // s_ij = dot(Q_i, k_j) * scale
            float dot_qk = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dot_qk += Qi[d] * k_regs[d];
            const float p_ij = expf(dot_qk * scale - lse_i);

            // dV_j += p_ij * dO_i
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dv_regs[d] += p_ij * dOi[d];

            // dp_ij = dot(dO_i, v_j)
            float dp_ij = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dp_ij += dOi[d] * v_regs[d];

            // ds_ij = p_ij * (dp_ij - D_i)
            const float ds_scaled = p_ij * (dp_ij - D_i) * scale;

            // dK_j += ds_scaled * Q_i
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dk_regs[d] += ds_scaled * Qi[d];
        }

        __syncthreads();
    }

    // Write dK_j, dV_j
    float* dK_ptr = dK + (bh * T_k + k_local) * HEAD_DIM;
    float* dV_ptr = dV + (bh * T_k + k_local) * HEAD_DIM;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) { dK_ptr[d] = dk_regs[d]; dV_ptr[d] = dv_regs[d]; }
}

// ---------------------------------------------------------------------------
// Generic backward kernels (non-templated, any D <= 256)
// ---------------------------------------------------------------------------
static constexpr int MAX_D_BWD = 256;

__global__ void flash_attn_bwd_dq_kernel_generic(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ LSE,
    float* __restrict__       dQ,
    float* __restrict__       D_buf,
    int T_q, int T_k, int D,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset)
{
    const int bh      = blockIdx.y;
    const int q_local = blockIdx.x * BLOCK_Q + threadIdx.x;
    if (q_local >= T_q) return;
    const int q_global = q_offset + q_local;

    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + BLOCK_K * D;

    const float* Q_ptr  = Q  + (bh * T_q + q_local) * D;
    const float* dO_ptr = dO + (bh * T_q + q_local) * D;
    const float* O_ptr  = O  + (bh * T_q + q_local) * D;

    float q_regs[MAX_D_BWD];
    float do_regs[MAX_D_BWD];
    float dq_regs[MAX_D_BWD];
    for (int d = 0; d < D; ++d) { q_regs[d] = Q_ptr[d]; do_regs[d] = dO_ptr[d]; dq_regs[d] = 0.0f; }

    float D_i = 0.0f;
    for (int d = 0; d < D; ++d) D_i += do_regs[d] * O_ptr[d];
    D_buf[bh * T_q + q_local] = D_i;

    const float lse_i = LSE[bh * T_q + q_local];
    const int num_k_blocks = (T_k + BLOCK_K - 1) / BLOCK_K;

    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_block_start  = kb * BLOCK_K;
        const int k_global_start = k_offset + k_block_start;
        if (is_causal && k_global_start > q_global) break;

        const int k_local_t = k_block_start + threadIdx.x;
        if (k_local_t < T_k) {
            const float* Kp = K + (bh * T_k + k_local_t) * D;
            const float* Vp = V + (bh * T_k + k_local_t) * D;
            float* Ks = K_tile + threadIdx.x * D;
            float* Vs = V_tile + threadIdx.x * D;
            for (int d = 0; d < D; ++d) { Ks[d] = Kp[d]; Vs[d] = Vp[d]; }
        } else {
            float* Ks = K_tile + threadIdx.x * D;
            float* Vs = V_tile + threadIdx.x * D;
            for (int d = 0; d < D; ++d) { Ks[d] = 0.0f; Vs[d] = 0.0f; }
        }
        __syncthreads();

        const int tile_size = (k_block_start + BLOCK_K <= T_k) ? BLOCK_K : (T_k - k_block_start);
        for (int j = 0; j < tile_size; ++j) {
            const int k_global_j = k_global_start + j;
            if (is_causal && k_global_j > q_global) continue;
            const float* Kj = K_tile + j * D;
            const float* Vj = V_tile + j * D;
            float dot_qk = 0.0f;
            for (int d = 0; d < D; ++d) dot_qk += q_regs[d] * Kj[d];
            const float p_ij = expf(dot_qk * scale - lse_i);
            float dp_ij = 0.0f;
            for (int d = 0; d < D; ++d) dp_ij += do_regs[d] * Vj[d];
            const float ds_scaled = p_ij * (dp_ij - D_i) * scale;
            for (int d = 0; d < D; ++d) dq_regs[d] += ds_scaled * Kj[d];
        }
        __syncthreads();
    }

    float* dQ_ptr = dQ + (bh * T_q + q_local) * D;
    for (int d = 0; d < D; ++d) dQ_ptr[d] = dq_regs[d];
}

__global__ void flash_attn_bwd_dkdv_kernel_generic(
    const float* __restrict__ Q,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D_buf,
    float* __restrict__       dK,
    float* __restrict__       dV,
    int T_q, int T_k, int D,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset,
    const float* __restrict__ K)
{
    const int bh      = blockIdx.y;
    const int k_local = blockIdx.x * BLOCK_K + threadIdx.x;
    if (k_local >= T_k) return;
    const int k_global = k_offset + k_local;

    // Shared: Q_tile [BLOCK_Q*D] + dO_tile [BLOCK_Q*D] + LSE_tile [BLOCK_Q] + D_tile [BLOCK_Q]
    extern __shared__ float smem[];
    float* Q_tile   = smem;
    float* dO_tile  = smem + BLOCK_Q * D;
    float* LSE_tile = smem + 2 * BLOCK_Q * D;
    float* D_tile   = LSE_tile + BLOCK_Q;

    const float* K_ptr = K + (bh * T_k + k_local) * D;
    const float* V_ptr = V + (bh * T_k + k_local) * D;

    float k_regs[MAX_D_BWD];
    float v_regs[MAX_D_BWD];
    float dk_regs[MAX_D_BWD];
    float dv_regs[MAX_D_BWD];
    for (int d = 0; d < D; ++d) { k_regs[d] = K_ptr[d]; v_regs[d] = V_ptr[d]; dk_regs[d] = 0.0f; dv_regs[d] = 0.0f; }

    const int num_q_blocks = (T_q + BLOCK_Q - 1) / BLOCK_Q;

    for (int qb = 0; qb < num_q_blocks; ++qb) {
        const int q_block_start  = qb * BLOCK_Q;
        const int q_global_start = q_offset + q_block_start;
        if (is_causal && (q_global_start + BLOCK_Q - 1) < k_global) continue;

        const int q_local_t = q_block_start + threadIdx.x;
        if (q_local_t < T_q) {
            const float* Qp  = Q  + (bh * T_q + q_local_t) * D;
            const float* dOp = dO + (bh * T_q + q_local_t) * D;
            float* Qt  = Q_tile  + threadIdx.x * D;
            float* dOt = dO_tile + threadIdx.x * D;
            for (int d = 0; d < D; ++d) { Qt[d] = Qp[d]; dOt[d] = dOp[d]; }
            LSE_tile[threadIdx.x] = LSE[bh * T_q + q_local_t];
            D_tile[threadIdx.x]   = D_buf[bh * T_q + q_local_t];
        } else {
            float* Qt  = Q_tile  + threadIdx.x * D;
            float* dOt = dO_tile + threadIdx.x * D;
            for (int d = 0; d < D; ++d) { Qt[d] = 0.0f; dOt[d] = 0.0f; }
            LSE_tile[threadIdx.x] = 1e30f;
            D_tile[threadIdx.x]   = 0.0f;
        }
        __syncthreads();

        const int tile_size = (q_block_start + BLOCK_Q <= T_q) ? BLOCK_Q : (T_q - q_block_start);
        for (int i = 0; i < tile_size; ++i) {
            const int q_global_i = q_global_start + i;
            if (is_causal && k_global > q_global_i) continue;
            const float* Qi  = Q_tile  + i * D;
            const float* dOi = dO_tile + i * D;
            const float  lse_i = LSE_tile[i];
            const float  Di    = D_tile[i];
            float dot_qk = 0.0f;
            for (int d = 0; d < D; ++d) dot_qk += Qi[d] * k_regs[d];
            const float p_ij = expf(dot_qk * scale - lse_i);
            for (int d = 0; d < D; ++d) dv_regs[d] += p_ij * dOi[d];
            float dp_ij = 0.0f;
            for (int d = 0; d < D; ++d) dp_ij += dOi[d] * v_regs[d];
            const float ds_scaled = p_ij * (dp_ij - Di) * scale;
            for (int d = 0; d < D; ++d) dk_regs[d] += ds_scaled * Qi[d];
        }
        __syncthreads();
    }

    float* dK_ptr = dK + (bh * T_k + k_local) * D;
    float* dV_ptr = dV + (bh * T_k + k_local) * D;
    for (int d = 0; d < D; ++d) { dK_ptr[d] = dk_regs[d]; dV_ptr[d] = dv_regs[d]; }
}

// ---------------------------------------------------------------------------
// launch_flash_attn_bwd_f32
// ---------------------------------------------------------------------------
void launch_flash_attn_bwd_f32(
    const float* Q,
    const float* K,
    const float* V,
    const float* O,
    const float* dO,
    const float* LSE,
    float*       dQ,
    float*       dK,
    float*       dV,
    float*       D_buf,
    int BH, int T_q, int T_k, int D,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset)
{
    // --- Kernel 1: dQ (also computes D_buf) ---
    {
        const dim3 grid((T_q + BLOCK_Q - 1) / BLOCK_Q, BH);
        const dim3 block(BLOCK_Q);
        const int  smem = 2 * BLOCK_K * D * sizeof(float);  // K_tile + V_tile

        switch (D) {
        case 32:
            flash_attn_bwd_dq_kernel<32><<<grid, block, smem>>>(
                Q, K, V, O, dO, LSE, dQ, D_buf,
                T_q, T_k, scale, is_causal, q_offset, k_offset);
            break;
        case 64:
            flash_attn_bwd_dq_kernel<64><<<grid, block, smem>>>(
                Q, K, V, O, dO, LSE, dQ, D_buf,
                T_q, T_k, scale, is_causal, q_offset, k_offset);
            break;
        case 128:
            flash_attn_bwd_dq_kernel<128><<<grid, block, smem>>>(
                Q, K, V, O, dO, LSE, dQ, D_buf,
                T_q, T_k, scale, is_causal, q_offset, k_offset);
            break;
        default:
            flash_attn_bwd_dq_kernel_generic<<<grid, block, smem>>>(
                Q, K, V, O, dO, LSE, dQ, D_buf,
                T_q, T_k, D, scale, is_causal, q_offset, k_offset);
            break;
        }
    }

    // --- Kernel 2: dK, dV (reads D_buf written by Kernel 1) ---
    // Same stream 0 -- sequential by default, D_buf is ready.
    {
        const dim3 grid((T_k + BLOCK_K - 1) / BLOCK_K, BH);
        const dim3 block(BLOCK_K);
        // Q_tile [BLOCK_Q*D] + dO_tile [BLOCK_Q*D] + LSE_tile [BLOCK_Q] + D_tile [BLOCK_Q]
        const int  smem = (2 * BLOCK_Q * D + 2 * BLOCK_Q) * sizeof(float);

        switch (D) {
        case 32:
            flash_attn_bwd_dkdv_kernel<32><<<grid, block, smem>>>(
                Q, V, dO, LSE, D_buf, dK, dV,
                T_q, T_k, scale, is_causal, q_offset, k_offset, K);
            break;
        case 64:
            flash_attn_bwd_dkdv_kernel<64><<<grid, block, smem>>>(
                Q, V, dO, LSE, D_buf, dK, dV,
                T_q, T_k, scale, is_causal, q_offset, k_offset, K);
            break;
        case 128:
            flash_attn_bwd_dkdv_kernel<128><<<grid, block, smem>>>(
                Q, V, dO, LSE, D_buf, dK, dV,
                T_q, T_k, scale, is_causal, q_offset, k_offset, K);
            break;
        default:
            flash_attn_bwd_dkdv_kernel_generic<<<grid, block, smem>>>(
                Q, V, dO, LSE, D_buf, dK, dV,
                T_q, T_k, D, scale, is_causal, q_offset, k_offset, K);
            break;
        }
    }
}
