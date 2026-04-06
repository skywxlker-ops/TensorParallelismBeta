#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <stdio.h>
#include "AttentionKernels.h"

namespace OwnTensor {

// ============================================================================
// Constants
// ============================================================================

static constexpr int MAX_HD = 256;   // max supported head dimension


// ============================================================================
// Warp-level and block-level reduction primitives
// ============================================================================

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float blockReduceMax(float val) {
    __shared__ float warp_vals[32];
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    val = warpReduceMax(val);
    if (laneId == 0) warp_vals[warpId] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float bmax = -INFINITY;
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < nw; ++w) bmax = fmaxf(bmax, warp_vals[w]);
        warp_vals[0] = bmax;
    }
    __syncthreads();
    return warp_vals[0];
}

__device__ float blockReduceSum(float val) {
    __shared__ float warp_vals[32];
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    val = warpReduceSum(val);
    if (laneId == 0) warp_vals[warpId] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float bsum = 0.0f;
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < nw; ++w) bsum += warp_vals[w];
        warp_vals[0] = bsum;
    }
    __syncthreads();
    return warp_vals[0];
}

// ============================================================================
// LSE Kernel
// ============================================================================

__global__ void compute_row_lse_kernel(
    const float* __restrict__ scores,
    float* __restrict__ lse,
    int64_t T,
    bool is_causal)
{
    const int64_t row = blockIdx.x;
    const int64_t qi  = row % T;
    const float* row_ptr = scores + row * T;
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    extern __shared__ float smem[];

    float local_max = -INFINITY;
    for (int j = tid; j < T; j += bdim) {
        float val = row_ptr[j];
        if (is_causal && j > qi) val = -INFINITY;
        local_max = fmaxf(local_max, val);
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float row_max = smem[0];

    float local_sum = 0.0f;
    for (int j = tid; j < T; j += bdim) {
        float val = row_ptr[j];
        if (is_causal && j > qi) val = -INFINITY;
        local_sum += expf(val - row_max);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0];

    if (tid == 0)
        lse[row] = (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
}


// ============================================================================
// Forward Kernel: Fused Memory-Efficient Attention (shared-memory-centric)
// ============================================================================
//
// Computes: O = softmax(Q @ K^T / sqrt(hd)) @ V
// WITHOUT materializing the full T x T attention matrix.
//
// Design: MULTIPLE threads per query row (mirrors the backward kernel)
//
//   Grid : (ceil(T / FWD_BQ), B * nh)
//   Block: FWD_NUM_THREADS = 256
//
//   Thread mapping: 256 threads = 32 rows x 8 threads/row
//     qi_local  = tid / 8    -- which query row (0..31)
//     local_tid = tid % 8    -- position within the 8-thread row group
//
//   Shared memory layout (maximized -- everything shared lives here,
//   just like the backward kernel):
//
//     s_q      [BQ x hd]    -- Q tile, loaded once, persistent
//     s_kv     [BK x hd]    -- K tile then V tile (reused each iter)
//     s_scores [BQ x BK]    -- score/attention-weight matrix
//     s_m      [BQ]          -- running max per query row
//     s_l      [BQ]          -- running sum per query row
//
//   Per thread:
//     reg_out[hd/8]  -- output accumulator in registers (not smem)
//                       Each of 8 threads owns a DIFFERENT slice of hd.
//
//   Why s_scores in shared memory?
//     During Score GEMM: each thread computes 4 dot products (split by
//     key columns). During P@V GEMM: each thread accumulates hd/8 output
//     dims, reading ALL BK attention weights. Thread roles CHANGE between
//     phases -- s_scores is the communication channel, exactly like
//     ds_buf/p_buf in the backward kernel.
//
//   Why s_m/s_l in shared memory?
//     Only local_tid==0 writes the updated max/sum. All 4 threads in the
//     row group read it for rescaling. Mirrors LSE/D in the backward.
//
//   Algorithm (online softmax):
//     1. Cooperative load Q tile -> s_q
//     2. For each K/V tile:
//        a. Cooperative load K tile -> s_kv
//        b. Score GEMM: each thread computes 4 dots -> s_scores
//        c. Online softmax via warp shuffle across 4-thread groups:
//           partial max -> shuffle reduce -> update s_m
//           rescale reg_out by exp(m_old - m_new)
//           exp(scores) -> s_scores, partial sum -> shuffle -> s_l
//        d. Cooperative load V tile -> s_kv
//        e. P@V GEMM: read all BK weights from s_scores,
//           accumulate hd/8 dims per thread in reg_out
//     3. O = reg_out / s_l
//     4. LSE = s_m + log(s_l)
//

// Forward tuning constants
static constexpr int FWD_BQ              = 32;  // query rows per block
static constexpr int FWD_BK              = 32;  // key rows per tile
static constexpr int FWD_THREADS_PER_ROW = 8;   // threads per Q row (was 4, now 8 for occupancy)
static constexpr int FWD_NUM_THREADS     = FWD_BQ * FWD_THREADS_PER_ROW;  // 256
static constexpr int FWD_KJ_PER_THREAD   = FWD_BK / FWD_THREADS_PER_ROW; // 4
static constexpr int FWD_SMEM_PAD        = 1;   // +1 padding to eliminate bank conflicts
static constexpr int FWD_SCORE_STRIDE    = FWD_BK + FWD_SMEM_PAD;  // 33 (odd → no conflict)

template<int HeadDim>
__global__ void fused_attn_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int64_t T,
    float scale, bool is_causal,
    float dropout_p, const float* __restrict__ dropout_mask)
{
    // All dimension-derived values are now compile-time constants
    constexpr uint HD_PAD       = HeadDim + FWD_SMEM_PAD;   // padded stride (e.g. 65)
    constexpr uint D_PER_THREAD = (HeadDim + FWD_THREADS_PER_ROW - 1) / FWD_THREADS_PER_ROW;

    const uint qi_block = (int64_t)blockIdx.x * FWD_BQ;
    const uint bnh      = blockIdx.y;
    const uint tid      = threadIdx.x;

    // ---- Thread-to-query-row mapping ----
    // 8 consecutive threads share one query row:
    //   threads 0-7     -> row 0
    //   threads 8-15    -> row 1
    //   ...
    //   threads 248-255 -> row 31
    //
    // NOTE: the warp_id/lane_id layout from the coalescing doc breaks the
    // warp shuffle reductions (they can't cross warp boundaries). The
    // PRACTICAL FIX in that doc keeps this mapping and stages output through
    // shared memory instead.
    const int qi_local  = tid / FWD_THREADS_PER_ROW;   // 0..31
    const int local_tid = tid % FWD_THREADS_PER_ROW;   // 0..7
    const int64_t qi_global = qi_block + qi_local;
    const bool qi_valid = (qi_global < T);

    // ---- Pointers for this batch*head ----
    const float* Q_bnh   = Q   + bnh * T * HeadDim;
    const float* K_bnh   = K   + bnh * T * HeadDim;
    const float* V_bnh   = V   + bnh * T * HeadDim;
    float*       O_bnh   = O   + bnh * T * HeadDim;
    float*       LSE_bnh = LSE + bnh * T;

    // ===================================================================
    // Shared memory layout (padded to eliminate bank conflicts)
    // ===================================================================
    //
    //   Row strides are padded by +1 so successive rows map to different
    //   banks (stride 65 and 33 are odd → conflict-free across rows).
    //
    //   +-------------------+-------------------+-----------------+-----+-----+-------------------+
    //   |  s_q              |  s_kv             |  s_scores       | s_m | s_l |  s_out            |
    //   |  [BQ x HD_PAD]    |  [BK x HD_PAD]    |  [BQ x (BK+1)] |[BQ] |[BQ] |  [BQ x HD_PAD]   |
    //   +-------------------+-------------------+-----------------+-----+-----+-------------------+
    //
    //   s_out: staging buffer for coalesced global writeback (same padded
    //          stride as s_q → same conflict-free access pattern).
    //
    extern __shared__ float smem[];
    float* s_q      = smem;                                              // [BQ x HD_PAD]
    float* s_kv     = s_q      + FWD_BQ * HD_PAD;                       // [BK x HD_PAD]
    float* s_scores = s_kv     + FWD_BK * HD_PAD;                       // [BQ x FWD_SCORE_STRIDE]
    float* s_m      = s_scores + FWD_BQ * FWD_SCORE_STRIDE;             // [BQ]
    float* s_l      = s_m      + FWD_BQ;                                // [BQ]
    float* s_out    = s_l      + FWD_BQ;                                // [BQ x HD_PAD]

    // ---- Initialize running max and sum in shared memory ----
    if (local_tid == 0) {
        s_m[qi_local] = -INFINITY;
        s_l[qi_local] = 0.0f;
    }

    // ---- Output accumulator in registers ----
    // HeadDim is now a compile-time constant, so the compiler knows
    // D_PER_THREAD exactly and keeps reg_out entirely in ACTUAL registers
    // (no spilling to local memory).
    // Strided assignment: thread local_tid owns dimensions {local_tid, local_tid+8, local_tid+16, ...}
    // i.e., reg_out[dd] accumulates output for dimension (local_tid + dd * FWD_THREADS_PER_ROW)

    float reg_out[D_PER_THREAD];   // compile-time size → guaranteed registers
    #pragma unroll
    for (int i = 0; i < D_PER_THREAD; i++) reg_out[i] = 0.0f;

    // ===================================================================
    // Step 1: Cooperative load Q tile into s_q (ALL 256 threads help)
    // ===================================================================
    {
        constexpr int64_t total = (int64_t)FWD_BQ * HeadDim;
        for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
            int q = (int)(i / HeadDim);
            int d = (int)(i % HeadDim);
            s_q[q * HD_PAD + d] = (qi_block + q < T) ? Q_bnh[(qi_block + q) * HeadDim + d] : 0.0f;
        }
    }
    __syncthreads();

    const int actual_q = ((int)(T - qi_block) < FWD_BQ)
                         ? (int)(T - qi_block) : FWD_BQ;
    if (actual_q <= 0) return;

    // ---- Causal early termination ----
    const int64_t max_kj = is_causal
        ? (((qi_block + actual_q) < T) ? (qi_block + actual_q) : T)
        : T;

    // ===================================================================
    // Step 2: Main loop over key/value tiles
    // ===================================================================
    for (int64_t kj_block = 0; kj_block < max_kj; kj_block += FWD_BK) {
        const int block_len = ((int)(T - kj_block) < FWD_BK)
                              ? (int)(T - kj_block) : FWD_BK;

        // ---------------------------------------------------------------
        // Step 2a: Cooperative load K tile into s_kv (padded stride)
        // ---------------------------------------------------------------
        {
            const int64_t total = (int64_t)block_len * HeadDim;
            for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
                int k = (int)(i / HeadDim);
                int d = (int)(i % HeadDim);
                s_kv[k * HD_PAD + d] = K_bnh[(kj_block + k) * HeadDim + d];
            }
        }
        __syncthreads();    // K tile ready in s_kv

        // ---------------------------------------------------------------
        // Step 2b: Score GEMM -- Q[BQ x hd] @ K[BK x hd]^T -> s_scores
        // ---------------------------------------------------------------
        {
            const int kj_start = local_tid * FWD_KJ_PER_THREAD;

            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                float dot;
                if (kj < block_len && qi_valid) {
                    dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HeadDim; d++)
                        dot += s_q[qi_local * HD_PAD + d] * s_kv[kj * HD_PAD + d];
                    dot *= scale;

                    // Causal mask
                    if (is_causal && (kj_block + kj) > qi_global)
                        dot = -INFINITY;
                } else {
                    dot = -INFINITY;
                }
                s_scores[qi_local * FWD_SCORE_STRIDE + kj] = dot;
            }
        }
        __syncthreads();    // all scores in s_scores

        // ---------------------------------------------------------------
        // Step 2c: Online softmax (warp shuffle across 8-thread groups)
        // ---------------------------------------------------------------
        {
            const int kj_start = local_tid * FWD_KJ_PER_THREAD;

            // (a) Partial row-max over this thread's 4 keys
            float partial_max = -INFINITY;
            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                if (kj < block_len)
                    partial_max = fmaxf(partial_max,
                                        s_scores[qi_local * FWD_SCORE_STRIDE + kj]);
            }

            // (b) Warp shuffle: reduce max across 8 threads in the group
            float row_max = partial_max;
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 1));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 2));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 4));

            // (c) Read old running max from shared memory
            float m_old = s_m[qi_local];
            float m_new = fmaxf(m_old, row_max);

            // (d) Correction factor for previous accumulations
            float alpha;
            if (m_old == -INFINITY)       alpha = 0.0f;
            else if (m_old == m_new)      alpha = 1.0f;
            else                          alpha = expf(m_old - m_new);

            // (e) Rescale this thread's output accumulator slice
            #pragma unroll
            for (int i = 0; i < D_PER_THREAD; i++)
                reg_out[i] *= alpha;

            // (f) Exponentiate scores, apply dropout, accumulate post-dropout sum.
            //
            // Step 2c is split so that s_l accumulates post-dropout weights
            // (walkthrough Option 2): compute exp → apply mask → sum masked values.
            // This ensures the final normalisation (reg_out / s_l) is correct
            // without any post-hoc correction of s_l.
            //
            // Dropout uses Option B (pre-generated mask tensor, shape [B*nh, T, T]):
            //   mask[bnh, qi, kj] already encodes the Bernoulli decision scaled by
            //   1/(1-p), so multiplying exp_s by the mask value both zeroes dropped
            //   entries and rescales surviving ones in one operation.
            float partial_sum = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                float exp_s;
                if (kj < block_len && m_new > -INFINITY) {
                    exp_s = expf(s_scores[qi_local * FWD_SCORE_STRIDE + kj] - m_new);
                    // Apply pre-generated dropout mask (scale-and-zero in one multiply).
                    // When dropout_p == 0 or no mask is provided this branch is skipped.
                    if (dropout_p > 0.0f && dropout_mask != nullptr) {
                        float m_val = dropout_mask[(bnh * T + qi_global) * T
                                                   + (kj_block + kj)];
                        exp_s *= m_val;
                    }
                } else {
                    exp_s = 0.0f;
                }
                s_scores[qi_local * FWD_SCORE_STRIDE + kj] = exp_s;
                partial_sum += exp_s;   // accumulate post-dropout sum → s_l correct
            }
            // Zero out padding
            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                if (kj >= block_len)
                    s_scores[qi_local * FWD_SCORE_STRIDE + kj] = 0.0f;
            }

            // (g) Warp shuffle: reduce sum across 8 threads
            float row_sum = partial_sum;
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1);
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 2);
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 4);

            // (h) Update shared running state (one writer per row)
            if (local_tid == 0) {
                s_l[qi_local] = alpha * s_l[qi_local] + row_sum;
                s_m[qi_local] = m_new;
            }
        }
        __syncthreads();    // s_scores holds P, s_m/s_l updated

        // ---------------------------------------------------------------
        // Step 2d: Cooperative load V tile into s_kv (overwrites K, padded)
        // ---------------------------------------------------------------
        {
            const int64_t total = (int64_t)block_len * HeadDim;
            for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
                int v = (int)(i / HeadDim);
                int d = (int)(i % HeadDim);
                s_kv[v * HD_PAD + d] = V_bnh[(kj_block + v) * HeadDim + d];
            }
        }
        __syncthreads();    // V tile ready in s_kv

        // ---------------------------------------------------------------
        // Step 2e: P@V GEMM -- accumulate into reg_out
        // ---------------------------------------------------------------
        // kj iterates to FWD_BK (compile-time constant = 32) instead of
        // block_len (runtime), so #pragma unroll 4 produces 8 fully-pipelined
        // groups of 4 smem reads, overlapping latency with compute.
        // Boundary guard is inside the kj loop; padded elements are 0 (set in
        // Step 2c so this is safe). s_scores rows were already zeroed for kj
        // >= block_len in the softmax step, so no extra cost.
        #pragma unroll
        for (int dd = 0; dd < D_PER_THREAD; dd++) {
            // Strided assignment: thread local_tid owns every 8th dimension.
            // Thread 0: d=0,8,16,...   Thread 1: d=1,9,17,...   Thread 7: d=7,15,23,...
            // For fixed kj: bank(thread_i) = (kj*HD_PAD + dd*8 + i) % 32 → all 8 banks distinct → 0 conflicts.
            int d = local_tid + dd * FWD_THREADS_PER_ROW;
            if (d < HeadDim) {
                float acc = 0.0f;
                #pragma unroll 4
                for (int kj = 0; kj < FWD_BK; kj++) {
                    if (kj < block_len)
                        acc += s_scores[qi_local * FWD_SCORE_STRIDE + kj]
                             * s_kv[kj * HD_PAD + d];
                }
                reg_out[dd] += acc;
            }
        }
        __syncthreads();    // before next iteration overwrites s_kv/s_scores
    }

    // ===================================================================
    // Step 3-4: Normalize into s_out (staged), then coalesced global write
    // ===================================================================
    //
    // WHY TWO PHASES:
    //   Direct write: thread i writes O[qi_global * HeadDim + d_start + dd].
    //   Threads 0..7 all write to the SAME row (row 0), not adjacent rows,
    //   so within a warp the 32 threads write to 4 different rows with
    //   stride HeadDim*4 bytes → separate cache-line transactions per thread.
    //
    //   Phase A (scatter to s_out): each thread writes its d-slice into the
    //   shared staging buffer at s_out[qi_local][d_start..d_end]. Same access
    //   pattern as loading s_q → already proven conflict-free with HD_PAD.
    //
    //   Phase B (coalesced gather to global): all 256 threads stride linearly
    //   over s_out and write to O. Adjacent threads → adjacent global addresses
    //   → full 128-byte cache-line utilization per transaction.

    // ---- Phase A: normalize and scatter to shared staging buffer ----
    if (qi_valid) {
        float li    = s_l[qi_local];
        float inv_l = (li > 0.0f) ? (1.0f / li) : 0.0f;

        #pragma unroll
        for (int dd = 0; dd < D_PER_THREAD; dd++) {
            int d = local_tid + dd * FWD_THREADS_PER_ROW;
            if (d < HeadDim)
                s_out[qi_local * HD_PAD + d] = reg_out[dd] * inv_l;
        }
    }
    __syncthreads();

    // ---- Phase B: cooperative coalesced writeback s_out -> O ----
    // Threads stride linearly: tid 0 writes (row0, d0), tid 1 writes (row0, d1), …
    // Adjacent tids → adjacent global memory addresses → coalesced.
    {
        const int64_t total = (int64_t)actual_q * HeadDim;
        for (int64_t i = (int64_t)tid; i < total; i += FWD_NUM_THREADS) {
            int q = (int)(i / HeadDim);
            int d = (int)(i % HeadDim);
            O_bnh[(qi_block + q) * HeadDim + d] = s_out[q * HD_PAD + d];
        }
    }

    // ---- Write LSE: one thread per row (local_tid == 0) ----
    if (local_tid == 0 && qi_valid) {
        float m = s_m[qi_local];
        float l = s_l[qi_local];
        LSE_bnh[qi_global] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
    }
}


// ============================================================================
// Forward launch function
// ============================================================================
//
// Shared memory (padded, with s_out staging buffer):
//   ((BQ + BK + BQ) * (hd + 1) + BQ * (BK + 1) + 2 * BQ) * sizeof(float)
//    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
//    s_q + s_kv + s_out, all with padded stride
//
//   hd=64:  (32+32+32)*65 + 32*33 + 64  =  7,360 floats =  29,440 bytes
//   hd=128: (32+32+32)*129 + 32*33 + 64 = 12,544 floats =  50,176 bytes (opt-in)
//   hd=256: (32+32+32)*257 + 32*33 + 64 = 24,768 floats =  99,072 bytes (opt-in)
//

static size_t compute_fwd_smem(int64_t hd) {
    int64_t hd_pad = hd + FWD_SMEM_PAD;
    return ((size_t)(FWD_BQ + FWD_BK + FWD_BQ) * hd_pad  // s_q + s_kv + s_out
          + (size_t)FWD_BQ * FWD_SCORE_STRIDE              // s_scores
          + (size_t)2 * FWD_BQ) * sizeof(float);           // s_m + s_l
}

static void launch_fwd_kernel(
    const float* Q, const float* K, const float* V,
    float* O, float* LSE,
    int64_t T, int64_t hd, float scale, bool is_causal,
    float dropout_p, const float* dropout_mask,
    int grid_y)
{
    // --- Query the device's shared memory limit ---
    int deviceId;
    cudaGetDevice(&deviceId);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

    size_t smem = compute_fwd_smem(hd);

    if ((int)smem > max_smem) {
        printf("fused_attn_forward: hd=%d needs %zu bytes smem, "
               "device max is %d. Cannot launch.\n",
               (int)hd, smem, max_smem);
        return;
    }

    // --- Grid: one block per query tile x one block per batch-head ---
    int grid_x = (int)((T + FWD_BQ - 1) / FWD_BQ);
    dim3 grid(grid_x, grid_y);

    //* print the grid and block dims
    // printf("The config for the kernel is: block{%d, 1} grid{%d, %d}\n", FWD_NUM_THREADS, grid.x, grid.y);
    // exit(1);

    // --- Template dispatch (same pattern as backward kernel) ---
    // HeadDim must be a compile-time constant so reg_out stays in registers.
#define LAUNCH_FWD(HD) \
    do { \
        auto* kernel = fused_attn_forward_kernel<HD>; \
        cudaFuncSetAttribute(kernel, \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, \
                             (int)smem); \
        kernel<<<grid, FWD_NUM_THREADS, smem>>>( \
            Q, K, V, O, LSE, T, scale, is_causal, dropout_p, dropout_mask); \
    } while (0)

    switch ((int)hd) {
        case   8: LAUNCH_FWD(  8); break;
        case  16: LAUNCH_FWD( 16); break;
        case  24: LAUNCH_FWD( 24); break;
        case  32: LAUNCH_FWD( 32); break;
        case  40: LAUNCH_FWD( 40); break;
        case  48: LAUNCH_FWD( 48); break;
        case  56: LAUNCH_FWD( 56); break;
        case  64: LAUNCH_FWD( 64); break;
        case  80: LAUNCH_FWD( 80); break;
        case  96: LAUNCH_FWD( 96); break;
        case 128: LAUNCH_FWD(128); break;
        case 160: LAUNCH_FWD(160); break;
        case 192: LAUNCH_FWD(192); break;
        case 256: LAUNCH_FWD(256); break;
        default:
            printf("fused_attn_forward: unsupported head_dim %d\n", (int)hd);
            break;
    }
#undef LAUNCH_FWD
}


// ============================================================================
// Exp1 Forward: WMMA TF32 Tensor-Core Score GEMM + P@V GEMM
// ============================================================================
//
// Changes vs fused_attn_forward_kernel (baseline scalar):
//   1. Score GEMM (Q @ K^T → s_scores): replaced scalar dot-product loop with
//      wmma::mma_sync using nvcuda::wmma::precision::tf32 (16×16×8 tiles).
//      Hardware converts FP32 → TF32 (10-bit mantissa) inside load_matrix_sync.
//      No explicit pre-conversion needed — pass float* directly.
//      Accumulator stays FP32; only the two input matrices are in TF32.
//   2. P@V GEMM (P @ V → s_pv): same WMMA TF32 approach.
//   3. Thread layout changed from "8 threads/row" to full 32-thread warps:
//      - Score GEMM: warps 0–3 each compute one 16×16 tile of s_scores[32×32]
//                    (2×2 grid of tiles). Warps 4–7 idle this phase.
//      - Softmax:    all 8 warps; each warp processes 4 rows (32/8).
//                    Each lane owns one key column (BK=32 → 32 lanes exactly).
//                    Warp-level shuffle reduction for row-max and row-sum.
//      - P@V GEMM:   all 8 warps; each warp handles tile_id = warp_id + pass*8
//                    in the 2×(hd/16) output tile grid.
//   4. Extra smem buffer s_pv[TC_BQ × HD_PAD] stores the P@V result before
//      accumulating into s_out, preventing the read-write race that would
//      occur if P@V output were stored directly back into s_kv while other
//      warps may still be reading V from s_kv.
//   5. s_out kept in smem (vs reg_out in registers for baseline). Required
//      because per-row alpha rescaling between KV tiles needs warp cooperation.
//   6. Only valid for HeadDim divisible by 16 (WMMA tile dimension WMMA_N=16).
//      launch_fwd_tc_kernel falls back to the scalar kernel for other head dims.
//
// Requires: Ampere GPU (sm_80+) for TF32 WMMA support.
//           The Makefile targets sm_86 so this is always satisfied.
//
// Shared memory layout:
//   s_q      [TC_BQ  × HD_PAD]   Q tile (fp32, loaded once per block)
//   s_kv     [TC_BK  × HD_PAD]   K tile then V tile (fp32, reused each KV iter)
//   s_scores [TC_BQ  × TC_BK]    raw dot-products → softmax weights P
//                                  (no padding; stride=TC_BK=32)
//   s_m      [TC_BQ]             running max per query row
//   s_l      [TC_BQ]             running denominator (sum of exp) per query row
//   s_out    [TC_BQ  × HD_PAD]   output accumulator (replaces reg_out)
//   s_pv     [TC_BQ  × HD_PAD]   temporary P@V result for this KV tile
//
// Grid/Block: identical to the scalar kernel.
//   Grid : (ceil(T / TC_BQ), B * nh)
//   Block: FWD_TC_NUM_THREADS = 256

static constexpr int FWD_TC_BQ          = 32;   // query rows per block
static constexpr int FWD_TC_BK          = 32;   // key/value rows per KV tile
static constexpr int FWD_TC_NUM_THREADS = 256;  // 8 warps × 32 lanes
static constexpr int FWD_TC_WMMA_M      = 16;   // WMMA tile rows
static constexpr int FWD_TC_WMMA_N      = 16;   // WMMA tile cols (HeadDim must be multiple)
static constexpr int FWD_TC_WMMA_K      = 8;    // WMMA tile K-dim for TF32
static constexpr int FWD_TC_SMEM_PAD    = 1;    // +1 padding to avoid bank conflicts

template<int HeadDim>
__global__ void fused_attn_forward_kernel_tc(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    float*       __restrict__ LSE,
    int64_t T,
    float scale, bool is_causal,
    float dropout_p, const float* __restrict__ dropout_mask,
    int q_offset, int k_offset)
{
    static_assert(HeadDim % FWD_TC_WMMA_N == 0,
                  "fused_attn_forward_kernel_tc: HeadDim must be divisible by 16");
    using namespace nvcuda;

    // ── Compile-time tile layout constants ───────────────────────────────────
    constexpr int HD_PAD        = HeadDim + FWD_TC_SMEM_PAD;
    // Score GEMM: Q[32×hd] @ K[32×hd]^T → s_scores[32×32]
    //   2×2 grid of 16×16 output tiles; K-dimension split into hd/8 k-tiles.
    constexpr int SCORE_N_TILES = FWD_TC_BK  / FWD_TC_WMMA_N;   // 2
    constexpr int SCORE_K_TILES = HeadDim    / FWD_TC_WMMA_K;   // hd/8
    // P@V GEMM: P[32×32] @ V[32×hd] → s_pv[32×hd]
    //   2×(hd/16) output tiles; K-dimension (=32) split into 4 k-tiles.
    constexpr int PV_N_TILES    = HeadDim    / FWD_TC_WMMA_N;   // hd/16
    constexpr int PV_K_TILES    = FWD_TC_BK  / FWD_TC_WMMA_K;  // 4
    constexpr int PV_TOTAL      = 2 * PV_N_TILES;               // total output tiles
    constexpr int PV_PASSES     = (PV_TOTAL + 7) / 8;           // passes over 8 warps
    constexpr int ROWS_PER_WARP = FWD_TC_BQ / (FWD_TC_NUM_THREADS / 32); // 4 rows/warp

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid     = threadIdx.x;

    const int64_t qi_block = (int64_t)blockIdx.x * FWD_TC_BQ;
    const int     bnh      = blockIdx.y;

    const float* Q_bnh   = Q   + bnh * T * HeadDim;
    const float* K_bnh   = K   + bnh * T * HeadDim;
    const float* V_bnh   = V   + bnh * T * HeadDim;
    float*       O_bnh   = O   + bnh * T * HeadDim;
    float*       LSE_bnh = LSE + bnh * T;

    // ── Shared memory layout ──────────────────────────────────────────────────
    extern __shared__ float smem[];
    float* s_q      = smem;
    float* s_kv     = s_q      + FWD_TC_BQ * HD_PAD;
    float* s_scores = s_kv     + FWD_TC_BK * HD_PAD;  // stride = TC_BK (no pad)
    float* s_m      = s_scores + FWD_TC_BQ * FWD_TC_BK;
    float* s_l      = s_m      + FWD_TC_BQ;
    float* s_out    = s_l      + FWD_TC_BQ;
    float* s_pv     = s_out    + FWD_TC_BQ * HD_PAD;  // P@V result (separate buffer)

    // ── Initialize running state and output accumulator ───────────────────────
    for (int i = tid; i < FWD_TC_BQ; i += FWD_TC_NUM_THREADS) {
        s_m[i] = -INFINITY;
        s_l[i] =  0.0f;
    }
    for (int i = tid; i < FWD_TC_BQ * HD_PAD; i += FWD_TC_NUM_THREADS) {
        s_out[i] = 0.0f;
    }

    // ── Step 1: Cooperative load Q tile → s_q ────────────────────────────────
    for (int i = tid; i < FWD_TC_BQ * HeadDim; i += FWD_TC_NUM_THREADS) {
        const int q = i / HeadDim;
        const int d = i % HeadDim;
        s_q[q * HD_PAD + d] = (qi_block + q < T)
                               ? Q_bnh[(qi_block + q) * HeadDim + d] : 0.0f;
    }
    __syncthreads();

    const int actual_q = (int)min((int64_t)FWD_TC_BQ, T - qi_block);
    if (actual_q <= 0) return;

    const int64_t max_kj = is_causal
        ? max((int64_t)0,
              min(q_offset + qi_block + (int64_t)actual_q - k_offset, T))
        : T;

    // ── Step 2: Main loop over KV tiles ──────────────────────────────────────
    for (int64_t kj_block = 0; kj_block < max_kj; kj_block += FWD_TC_BK) {
        const int block_len = (int)min((int64_t)FWD_TC_BK, T - kj_block);

        // ── 2a: Load K tile → s_kv ────────────────────────────────────────────
        for (int i = tid; i < FWD_TC_BK * HeadDim; i += FWD_TC_NUM_THREADS) {
            const int k = i / HeadDim;
            const int d = i % HeadDim;
            const int g = (int)kj_block + k;
            s_kv[k * HD_PAD + d] = (g < T) ? K_bnh[g * HeadDim + d] : 0.0f;
        }
        __syncthreads();    // K tile visible to Score GEMM

        // ── 2b: Score GEMM via WMMA TF32 — warps 0–3 ─────────────────────────
        //
        //   Computes Q[32×hd] @ K[32×hd]^T → s_scores[32×32].
        //   4 output tiles (2×2 of 16×16), one per warp:
        //     warp w → m_tile = w/2, n_tile = w%2
        //
        //   A (row_major): Q slice [m_tile*16 : (m_tile+1)*16][k*8 : (k+1)*8]
        //     load_matrix_sync(float*) converts FP32→TF32 internally.
        //
        //   B (col_major): K slice [n_tile*16 : (n_tile+1)*16][k*8 : (k+1)*8]
        //     col_major of a row-major K block gives K^T:
        //       B[k'][n'] = K_smem[(n_tile*16+n') * HD_PAD + k*8+k']
        //     ⟹ mma computes A × B = Q[m_tile rows] @ K[n_tile rows]^T  ✓
        //
        //   Accumulator: FP32 throughout.
        if (warp_id < 4) {
            const int m_tile = warp_id / SCORE_N_TILES;  // 0 or 1
            const int n_tile = warp_id % SCORE_N_TILES;  // 0 or 1

            wmma::fragment<wmma::accumulator,
                           FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K,
                           float> acc;
            wmma::fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int k = 0; k < SCORE_K_TILES; ++k) {
                wmma::fragment<wmma::matrix_a,
                               FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K,
                               wmma::precision::tf32, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b,
                               FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K,
                               wmma::precision::tf32, wmma::col_major> b_frag;

                wmma::load_matrix_sync(
                    a_frag,
                    s_q + m_tile * FWD_TC_WMMA_M * HD_PAD + k * FWD_TC_WMMA_K,
                    HD_PAD);

                wmma::load_matrix_sync(
                    b_frag,
                    s_kv + n_tile * FWD_TC_WMMA_N * HD_PAD + k * FWD_TC_WMMA_K,
                    HD_PAD);

                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            // Store raw dot-products to s_scores, stride = TC_BK = 32
            wmma::store_matrix_sync(
                s_scores + m_tile * FWD_TC_WMMA_M * FWD_TC_BK
                         + n_tile * FWD_TC_WMMA_N,
                acc, FWD_TC_BK, wmma::mem_row_major);
        }
        __syncthreads();    // s_scores ready for softmax (all warps)

        // ── 2c: Online softmax — scalar, all 8 warps ──────────────────────────
        //
        //   Thread layout: warp w handles rows [w*4 : w*4+4).
        //   Lane j handles key column j (BK=32 → one column per lane).
        //   Full row-max and row-sum reductions fit in a single warp shuffle.
        //
        //   Per row:
        //     (i)  Read scaled score or -INF (out-of-bounds / causal mask)
        //     (ii) Warp-reduce max → m_new; compute alpha = exp(m_old - m_new)
        //     (iii) Rescale s_out[row][*] by alpha (multi-pass over HeadDim)
        //     (iv) Exponentiate, apply dropout → P in s_scores
        //     (v)  Warp-reduce sum; update s_m[row], s_l[row] (lane 0 writes)
        {
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const int   row        = warp_id * ROWS_PER_WARP + r;
                const int64_t qi_global = qi_block + row;
                const bool  qi_valid   = (qi_global < T);

                // (i) Read score; apply scale and boundary / causal mask
                float val;
                if (lane_id < block_len && qi_valid) {
                    val = s_scores[row * FWD_TC_BK + lane_id] * scale;
                    if (is_causal && (k_offset + kj_block + lane_id) > (q_offset + qi_global))
                        val = -INFINITY;
                } else {
                    val = -INFINITY;
                }

                // (ii) Warp-reduce max
                float row_max = val;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));

                const float m_old = s_m[row];
                const float m_new = qi_valid ? fmaxf(m_old, row_max) : m_old;

                float alpha;
                if      (m_old == -INFINITY) alpha = 0.0f;
                else if (m_old == m_new)     alpha = 1.0f;
                else                         alpha = expf(m_old - m_new);

                // (iii) Rescale s_out[row][*] by alpha
                //   32 lanes cover HeadDim in ceil(HeadDim/32) passes
                for (int d = lane_id; d < HeadDim; d += 32)
                    s_out[row * HD_PAD + d] *= alpha;

                // (iv) Exponentiate and apply dropout mask
                float exp_s;
                if (lane_id < block_len && qi_valid && m_new > -INFINITY) {
                    exp_s = expf(val - m_new);
                    if (dropout_p > 0.0f && dropout_mask != nullptr)
                        exp_s *= dropout_mask[(bnh * T + qi_global) * T
                                              + (kj_block + lane_id)];
                } else {
                    exp_s = 0.0f;
                }
                s_scores[row * FWD_TC_BK + lane_id] = exp_s;  // P stored back

                // (v) Warp-reduce sum; update running state (lane 0 only)
                float row_sum = exp_s;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);

                if (lane_id == 0) {
                    s_l[row] = alpha * s_l[row] + row_sum;
                    s_m[row] = m_new;
                }
            }
        }
        __syncthreads();    // s_scores (P), s_m, s_l fully updated

        // ── 2d: Load V tile → s_kv (overwrites K) ────────────────────────────
        for (int i = tid; i < FWD_TC_BK * HeadDim; i += FWD_TC_NUM_THREADS) {
            const int v = i / HeadDim;
            const int d = i % HeadDim;
            const int g = (int)kj_block + v;
            s_kv[v * HD_PAD + d] = (g < T) ? V_bnh[g * HeadDim + d] : 0.0f;
        }
        __syncthreads();    // V tile ready for P@V GEMM

        // ── 2e: P@V GEMM via WMMA TF32 — all 8 warps ─────────────────────────
        //
        //   Computes P[32×32] @ V[32×hd] → s_pv[32×hd].
        //   Output grid: PV_M_TILES(=2) × PV_N_TILES(=hd/16) tiles of 16×16.
        //   Total tiles: PV_TOTAL = 2*(hd/16).  8 warps cover them in PV_PASSES.
        //   tile_id = warp_id + pass*8; m_tile = tile_id/PV_N_TILES.
        //
        //   A (row_major): P slice from s_scores, stride = TC_BK = 32.
        //   B (row_major): V slice from s_kv, stride = HD_PAD.
        //   Result → s_pv (separate buffer; avoids race with s_kv reads above).
        //
        //   Race analysis: All warps read s_kv (V) during their WMMA loops.
        //   All warps write to s_pv (a DIFFERENT buffer). No read-write conflict.
        //   The __syncthreads() after ensures s_pv is fully visible before step 2f.
        {
            for (int pass = 0; pass < PV_PASSES; ++pass) {
                const int tile_id = warp_id + pass * (FWD_TC_NUM_THREADS / 32);
                if (tile_id < PV_TOTAL) {
                    const int m_tile = tile_id / PV_N_TILES;
                    const int n_tile = tile_id % PV_N_TILES;

                    wmma::fragment<wmma::accumulator,
                                   FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K,
                                   float> acc;
                    wmma::fill_fragment(acc, 0.0f);

                    #pragma unroll
                    for (int k = 0; k < PV_K_TILES; ++k) {
                        wmma::fragment<wmma::matrix_a,
                                       FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K,
                                       wmma::precision::tf32, wmma::row_major> a_frag;
                        wmma::fragment<wmma::matrix_b,
                                       FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K,
                                       wmma::precision::tf32, wmma::row_major> b_frag;

                        // A: P[m_tile*16 : ...][k*8 : ...], row-major, stride=TC_BK
                        wmma::load_matrix_sync(
                            a_frag,
                            s_scores + m_tile * FWD_TC_WMMA_M * FWD_TC_BK
                                     + k    * FWD_TC_WMMA_K,
                            FWD_TC_BK);

                        // B: V[k*8 : ...][n_tile*16 : ...], row-major, stride=HD_PAD
                        wmma::load_matrix_sync(
                            b_frag,
                            s_kv + k * FWD_TC_WMMA_K * HD_PAD
                                 + n_tile * FWD_TC_WMMA_N,
                            HD_PAD);

                        wmma::mma_sync(acc, a_frag, b_frag, acc);
                    }

                    // Store P@V tile to s_pv (stride = HD_PAD, row-major)
                    wmma::store_matrix_sync(
                        s_pv + m_tile * FWD_TC_WMMA_M * HD_PAD
                             + n_tile * FWD_TC_WMMA_N,
                        acc, HD_PAD, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();    // all P@V tiles written to s_pv

        // ── 2f: Accumulate s_pv → s_out ──────────────────────────────────────
        for (int i = tid; i < FWD_TC_BQ * HeadDim; i += FWD_TC_NUM_THREADS) {
            const int q = i / HeadDim;
            const int d = i % HeadDim;
            s_out[q * HD_PAD + d] += s_pv[q * HD_PAD + d];
        }
        __syncthreads();    // before next KV tile overwrites s_kv / s_scores
    }

    // ── Step 3: Normalize s_out by s_l (in-place) ────────────────────────────
    for (int i = tid; i < actual_q * HeadDim; i += FWD_TC_NUM_THREADS) {
        const int   q    = i / HeadDim;
        const int   d    = i % HeadDim;
        const float li   = s_l[q];
        s_out[q * HD_PAD + d] *= (li > 0.0f) ? (1.0f / li) : 0.0f;
    }
    __syncthreads();

    // ── Step 4: Coalesced writeback s_out → global O ─────────────────────────
    for (int i = tid; i < actual_q * HeadDim; i += FWD_TC_NUM_THREADS) {
        const int q = i / HeadDim;
        const int d = i % HeadDim;
        O_bnh[(qi_block + q) * HeadDim + d] = s_out[q * HD_PAD + d];
    }

    // ── Step 5: Write LSE ─────────────────────────────────────────────────────
    for (int i = tid; i < actual_q; i += FWD_TC_NUM_THREADS) {
        const float m = s_m[i];
        const float l = s_l[i];
        LSE_bnh[qi_block + i] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
    }
}

// ============================================================================
// TC Forward: Shared memory size
// ============================================================================
//   s_q + s_kv + s_out + s_pv : 4 × [TC_BQ × HD_PAD]
//   s_scores                  : [TC_BQ × TC_BK] (no padding)
//   s_m + s_l                 : 2 × TC_BQ
static size_t compute_fwd_tc_smem(int64_t hd) {
    const size_t hd_pad = (size_t)hd + FWD_TC_SMEM_PAD;
    return (4ULL * FWD_TC_BQ * hd_pad          // s_q, s_kv, s_out, s_pv
          + (size_t)FWD_TC_BQ * FWD_TC_BK      // s_scores
          + 2ULL * FWD_TC_BQ)                  // s_m, s_l
         * sizeof(float);
}

// ============================================================================
// TC Forward: Launch function
// ============================================================================
//   Falls back to the scalar kernel for HeadDim not divisible by 16 (WMMA_N=16)
//   or if the required smem exceeds the device limit.
static void launch_fwd_tc_kernel(
    const float* Q, const float* K, const float* V,
    float* O, float* LSE,
    int64_t T, int64_t hd, float scale, bool is_causal,
    float dropout_p, const float* dropout_mask,
    int grid_y,
    int q_offset = 0, int k_offset = 0)
{
    if (hd % FWD_TC_WMMA_N != 0) {
        launch_fwd_kernel(Q, K, V, O, LSE, T, hd, scale, is_causal,
                          dropout_p, dropout_mask, grid_y);
        return;
    }

    int deviceId;
    cudaGetDevice(&deviceId);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

    const size_t smem = compute_fwd_tc_smem(hd);
    if ((int)smem > max_smem) {
        printf("fused_attn_forward_tc: hd=%d needs %zu bytes smem, "
               "device max is %d. Falling back to scalar kernel.\n",
               (int)hd, (size_t)smem, max_smem);
        launch_fwd_kernel(Q, K, V, O, LSE, T, hd, scale, is_causal,
                          dropout_p, dropout_mask, grid_y);
        return;
    }

    const int grid_x = (int)((T + FWD_TC_BQ - 1) / FWD_TC_BQ);
    const dim3 grid(grid_x, grid_y);

#define LAUNCH_FWD_TC(HD) \
    do { \
        auto* kernel = fused_attn_forward_kernel_tc<HD>; \
        cudaFuncSetAttribute(kernel, \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, \
                             (int)smem); \
        kernel<<<grid, FWD_TC_NUM_THREADS, smem>>>( \
            Q, K, V, O, LSE, T, scale, is_causal, dropout_p, dropout_mask, \
            q_offset, k_offset); \
    } while (0)

    switch ((int)hd) {
        case  16: LAUNCH_FWD_TC( 16); break;
        case  32: LAUNCH_FWD_TC( 32); break;
        case  48: LAUNCH_FWD_TC( 48); break;
        case  64: LAUNCH_FWD_TC( 64); break;
        case  80: LAUNCH_FWD_TC( 80); break;
        case  96: LAUNCH_FWD_TC( 96); break;
        case 128: LAUNCH_FWD_TC(128); break;
        case 160: LAUNCH_FWD_TC(160); break;
        case 192: LAUNCH_FWD_TC(192); break;
        case 256: LAUNCH_FWD_TC(256); break;
        default:
            launch_fwd_kernel(Q, K, V, O, LSE, T, hd, scale, is_causal,
                              dropout_p, dropout_mask, grid_y);
            break;
    }
#undef LAUNCH_FWD_TC
}


// ============================================================================
// Backward Pass
// ============================================================================

static constexpr int BWD_BLOCK_M   = 8;
static constexpr int BWD_WARP_SZ   = 32;
static constexpr int BWD_NUM_THREADS = BWD_BLOCK_M * BWD_WARP_SZ;  // 256

__inline__ __device__ float bwd_warp_sum(float val) {
    #pragma unroll
    for (int offset = BWD_WARP_SZ / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

struct MemEfficientBwdParams {
    const float* __restrict__ Q;
    const float* __restrict__ K;
    const float* __restrict__ V;
    const float* __restrict__ dO;
    const float* __restrict__ LSE;
    const float* __restrict__ D;
    float* __restrict__ dQ;
    float* __restrict__ dK;
    float* __restrict__ dV;
    int T;
    float scale;
    bool is_causal;
    int q_offset;
    int k_offset;
};

// Kernel 1: D[i] = dot(dO[i], O[i])
template<int HeadDim>
__global__ void mem_efficient_bwd_precompute_D(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    int T)
{
    constexpr int LocalN = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    const int bh      = blockIdx.y;
    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;
    const int row     = blockIdx.x * BWD_BLOCK_M + warp_id;
    if (row >= T) return;

    const long long off = (long long)bh * T * HeadDim + row * HeadDim;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < LocalN; ++i) {
        int k = lane_id + i * BWD_WARP_SZ;
        if (k < HeadDim)
            sum += dO[off + k] * O[off + k];
    }
    sum = bwd_warp_sum(sum);
    if (lane_id == 0)
        D[(long long)bh * T + row] = sum;
}

// Kernel 2: Unified backward (KV-tile-centric)
template<int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel(MemEfficientBwdParams params)
{
    constexpr int BlockN = (HeadDim < 64) ? 32 : (2048 / HeadDim);
    constexpr int LocalN = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;

    extern __shared__ float smem[];
    float* Ks     = smem;
    float* Vs     = Ks     + BlockN * HeadDim;
    float* dKs    = Vs     + BlockN * HeadDim;
    float* dVs    = dKs    + BlockN * HeadDim;
    float* ds_buf = dVs    + BlockN * HeadDim;
    float* p_buf  = ds_buf + BWD_BLOCK_M * BlockN;
    float* Q_buf  = p_buf  + BWD_BLOCK_M * BlockN;
    float* dO_buf = Q_buf  + BWD_BLOCK_M * HeadDim;

    const int bh         = blockIdx.y;
    const int tile_start = blockIdx.x * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;

    const float* Q_bh   = params.Q   + bh_off;
    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* dO_bh  = params.dO  + bh_off;
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Load K, V tiles; zero dKs, dVs
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        int r = idx / HeadDim;
        int g_row = tile_start + r;
        Ks [idx] = (g_row < params.T) ? K_bh[g_row * HeadDim + (idx % HeadDim)] : 0.0f;
        Vs [idx] = (g_row < params.T) ? V_bh[g_row * HeadDim + (idx % HeadDim)] : 0.0f;
        dKs[idx] = 0.0f;
        dVs[idx] = 0.0f;
    }
    __syncthreads();

    int q_start = Causal ? tile_start : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
        int qi = q_base + warp_id;
        bool valid = (qi < params.T);

        float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
        float L_qi = 0.0f, D_qi = 0.0f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            int k = lane_id + i * BWD_WARP_SZ;
            float qv = 0.0f, dov = 0.0f;
            if (valid && k < HeadDim) {
                qv  = Q_bh [qi * HeadDim + k];
                dov = dO_bh[qi * HeadDim + k];
            }
            q_local[i]  = qv;
            do_local[i] = dov;
            dq_local[i] = 0.0f;
            if (k < HeadDim) {
                Q_buf [warp_id * HeadDim + k] = qv;
                dO_buf[warp_id * HeadDim + k] = dov;
            }
        }
        if (valid) {
            L_qi = LSE_bh[qi];
            D_qi = D_bh[qi];
        }

        // Phase A: per-warp ds/p computation + dQ accumulation
        for (int j = 0; j < tile_size; ++j) {
            float dot_qk = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    dot_qk += q_local[i] * Ks[j * HeadDim + k];
            }
            float s = bwd_warp_sum(dot_qk) * params.scale;

            float p;
            if (!valid)                             p = 0.0f;
            else if (Causal && (tile_start+j) > qi) p = 0.0f;
            else                                    p = __expf(s - L_qi);

            float dot_dov = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    dot_dov += do_local[i] * Vs[j * HeadDim + k];
            }
            float dpv = bwd_warp_sum(dot_dov);
            float ds = p * (dpv - D_qi);

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    dq_local[i] += ds * params.scale * Ks[j * HeadDim + k];
            }

            if (lane_id == 0) {
                ds_buf[warp_id * BlockN + j] = ds;
                p_buf [warp_id * BlockN + j] = p;
            }
        }

        __syncthreads();

        // Phase B: cooperative dK/dV update
        for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
            int j = idx / HeadDim;
            int k = idx % HeadDim;
            float dk_acc = 0.0f, dv_acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < BWD_BLOCK_M; ++w) {
                dk_acc += ds_buf[w * BlockN + j] * Q_buf [w * HeadDim + k];
                dv_acc += p_buf [w * BlockN + j] * dO_buf[w * HeadDim + k];
            }
            dKs[idx] += dk_acc * params.scale;
            dVs[idx] += dv_acc;
        }

        __syncthreads();

        // Write dQ (atomicAdd)
        if (valid) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    atomicAdd(&dQ_bh[qi * HeadDim + k], dq_local[i]);
            }
        }
    }

    __syncthreads();

    // Write dK, dV to global
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        int r = idx / HeadDim;
        dK_bh[(tile_start + r) * HeadDim + (idx % HeadDim)] = dKs[idx];
        dV_bh[(tile_start + r) * HeadDim + (idx % HeadDim)] = dVs[idx];
    }
}
// ============================================================================
// Exp1: Fused Phase A+B via smem atomics + HD_PAD bank-conflict elimination
// ============================================================================
//
// Changes vs original:
//   - ds_buf, p_buf, Q_buf, dO_buf eliminated from smem (saves 2*BM*BlockN +
//     2*BM*HD floats per block)
//   - Padded smem stride (HD_PAD = HeadDim+1) on K/V/dK/dV tiles — same trick
//     as forward v2 to break all bank conflicts in the inner j-dot loop
//   - Each warp immediately atomicAdds its ds/p contributions into dKs/dVs
//     in shared memory (no Phase B barrier or staging loop)
//   - __syncthreads() only once per q_base iteration (after all warps done
//     with j-loop) to ensure dKs/dVs visibility before next q_base
//
// smem = 4 * BlockN * (HeadDim+1) * sizeof(float)

template<int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp1(MemEfficientBwdParams params)
{
    constexpr int BlockN  = (HeadDim < 64) ? 32 : (2048 / HeadDim);
    constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD  = HeadDim + 1;   // eliminates bank conflicts (borrowed from fwd v2)
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    extern __shared__ float smem[];
    float* Ks  = smem;
    float* Vs  = Ks  + BlockN * HD_PAD;
    float* dKs = Vs  + BlockN * HD_PAD;
    float* dVs = dKs + BlockN * HD_PAD;

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;

    const float* Q_bh   = params.Q   + bh_off;
    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* dO_bh  = params.dO  + bh_off;
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Cooperative load K/V tile into padded smem; zero dKs, dVs
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        int r = idx / HeadDim;
        int d = idx % HeadDim;
        int g_row = tile_start + r;
        float kv = (g_row < params.T) ? K_bh[g_row * HeadDim + d] : 0.0f;
        float vv = (g_row < params.T) ? V_bh[g_row * HeadDim + d] : 0.0f;
        Ks [r * HD_PAD + d] = kv;
        Vs [r * HD_PAD + d] = vv;
        dKs[r * HD_PAD + d] = 0.0f;
        dVs[r * HD_PAD + d] = 0.0f;
    }
    __syncthreads();

    const int q_start = Causal ? tile_start : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
        const int qi    = q_base + warp_id;
        const bool valid = (qi < params.T);

        float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
        float L_qi = 0.0f, D_qi = 0.0f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            int k = lane_id + i * BWD_WARP_SZ;
            float qv = 0.0f, dov = 0.0f;
            if (valid && k < HeadDim) {
                qv  = Q_bh [qi * HeadDim + k];
                dov = dO_bh[qi * HeadDim + k];
            }
            q_local[i]  = qv;
            do_local[i] = dov;
            dq_local[i] = 0.0f;
        }
        if (valid) { L_qi = LSE_bh[qi]; D_qi = D_bh[qi]; }

        // Fused A+B: per j, compute ds/p and immediately atomicAdd into dKs/dVs
        for (int j = 0; j < tile_size; ++j) {

            float dot_qk = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) dot_qk += q_local[i] * Ks[j * HD_PAD + k];
            }
            const float s = bwd_warp_sum(dot_qk) * params.scale;

            float p;
            if (!valid || (Causal && (tile_start + j) > qi))
                p = 0.0f;
            else
                p = exp2f(BWD_LOG2E * (s - L_qi));

            float dot_dov = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) dot_dov += do_local[i] * Vs[j * HD_PAD + k];
            }
            const float ds = p * (bwd_warp_sum(dot_dov) - D_qi);

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dq_local[i] += ds * params.scale * Ks[j * HD_PAD + k];
                    atomicAdd(&dKs[j * HD_PAD + k], ds * params.scale * q_local[i]);
                    atomicAdd(&dVs[j * HD_PAD + k], p  *                do_local[i]);
                }
            }
        } // j loop

        // Ensure all warp atomics visible in dKs/dVs before next q_base
        __syncthreads();

        if (valid) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    atomicAdd(&dQ_bh[qi * HeadDim + k], dq_local[i]);
            }
        }
    } // q_base loop

    __syncthreads();

    // Write dK, dV (de-padded)
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        int r = idx / HeadDim;
        int d = idx % HeadDim;
        dK_bh[(tile_start + r) * HeadDim + d] = dKs[r * HD_PAD + d];
        dV_bh[(tile_start + r) * HeadDim + d] = dVs[r * HD_PAD + d];
    }
}

// ============================================================================
// Exp2: Float4 vectorized K/V tile loads + padded smem + unrolled Phase B
// ============================================================================
//
// Changes vs original:
//   - HD_PAD stride on all smem tiles (bank-conflict free, same as fwd v2)
//   - float4 vectorized cooperative loads for K/V tile (4x fewer transactions)
//   - float4 vectorized dK/dV global writes
//   - Phase B w-loop fully unrolled at compile time (#pragma unroll with
//     compile-time constant BWD_BLOCK_M=8 — exposes ILP across 8 accumulators)
//   - ds_buf/p_buf/Q_buf/dO_buf kept (staging preserved, isolates bandwidth gain)
//
// smem = 4*BlockN*HD_PAD + 2*BM*BlockN + 2*BM*HD_PAD

template<int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp2(MemEfficientBwdParams params)
{
    constexpr int BlockN  = (HeadDim < 64) ? 32 : (2048 / HeadDim);
    constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD  = HeadDim + 1;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    extern __shared__ float smem[];
    float* Ks     = smem;
    float* Vs     = Ks     + BlockN  * HD_PAD;
    float* dKs    = Vs     + BlockN  * HD_PAD;
    float* dVs    = dKs    + BlockN  * HD_PAD;
    float* ds_buf = dVs    + BlockN  * HD_PAD;
    float* p_buf  = ds_buf + BWD_BLOCK_M * BlockN;
    float* Q_buf  = p_buf  + BWD_BLOCK_M * BlockN;
    float* dO_buf = Q_buf  + BWD_BLOCK_M * HD_PAD;

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;

    const float* Q_bh   = params.Q   + bh_off;
    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* dO_bh  = params.dO  + bh_off;
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Float4 vectorized K/V load into padded smem (HeadDim % 4 == 0 for all cases)
    // Scalar fallback for the +1 padding column (d == HeadDim never written by float4)
    {
        const int stride4 = HeadDim / 4;   // float4s per row (un-padded)
        for (int idx = threadIdx.x; idx < BlockN * stride4; idx += blockDim.x) {
            int r = idx / stride4;
            int d4 = idx % stride4;        // which float4 within a row
            int g_row = tile_start + r;
            float4 kv4 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 vv4 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (g_row < params.T) {
                kv4 = reinterpret_cast<const float4*>(K_bh + g_row * HeadDim)[d4];
                vv4 = reinterpret_cast<const float4*>(V_bh + g_row * HeadDim)[d4];
            }
            int base = r * HD_PAD + d4 * 4;
            Ks[base]   = kv4.x; Ks[base+1] = kv4.y;
            Ks[base+2] = kv4.z; Ks[base+3] = kv4.w;
            Vs[base]   = vv4.x; Vs[base+1] = vv4.y;
            Vs[base+2] = vv4.z; Vs[base+3] = vv4.w;
        }
        // zero dKs, dVs (scalar — padding slot included)
        for (int idx = threadIdx.x; idx < BlockN * HD_PAD; idx += blockDim.x) {
            dKs[idx] = 0.0f;
            dVs[idx] = 0.0f;
        }
    }
    __syncthreads();

    const int q_start = Causal ? tile_start : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
        const int qi    = q_base + warp_id;
        const bool valid = (qi < params.T);

        float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
        float L_qi = 0.0f, D_qi = 0.0f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            int k = lane_id + i * BWD_WARP_SZ;
            float qv = 0.0f, dov = 0.0f;
            if (valid && k < HeadDim) {
                qv  = Q_bh [qi * HeadDim + k];
                dov = dO_bh[qi * HeadDim + k];
            }
            q_local[i]  = qv;
            do_local[i] = dov;
            dq_local[i] = 0.0f;
            if (k < HeadDim) {
                Q_buf [warp_id * HD_PAD + k] = qv;
                dO_buf[warp_id * HD_PAD + k] = dov;
            }
        }
        if (valid) { L_qi = LSE_bh[qi]; D_qi = D_bh[qi]; }

        // Phase A (unchanged except padded smem access)
        for (int j = 0; j < tile_size; ++j) {

            float dot_qk = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) dot_qk += q_local[i] * Ks[j * HD_PAD + k];
            }
            const float s = bwd_warp_sum(dot_qk) * params.scale;

            float p;
            if (!valid || (Causal && (tile_start + j) > qi))
                p = 0.0f;
            else
                p = exp2f(BWD_LOG2E * (s - L_qi));

            float dot_dov = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) dot_dov += do_local[i] * Vs[j * HD_PAD + k];
            }
            const float ds = p * (bwd_warp_sum(dot_dov) - D_qi);

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) dq_local[i] += ds * params.scale * Ks[j * HD_PAD + k];
            }
            if (lane_id == 0) {
                ds_buf[warp_id * BlockN + j] = ds;
                p_buf [warp_id * BlockN + j] = p;
            }
        }

        __syncthreads();

        // Phase B: fully unrolled over BWD_BLOCK_M (compile-time = 8)
        for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
            const int j = idx / HeadDim;
            const int k = idx % HeadDim;
            float dk_acc = 0.0f, dv_acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < BWD_BLOCK_M; ++w) {
                dk_acc += ds_buf[w * BlockN + j] * Q_buf [w * HD_PAD + k];
                dv_acc += p_buf [w * BlockN + j] * dO_buf[w * HD_PAD + k];
            }
            dKs[j * HD_PAD + k] += dk_acc * params.scale;
            dVs[j * HD_PAD + k] += dv_acc;
        }

        __syncthreads();

        if (valid) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    atomicAdd(&dQ_bh[qi * HeadDim + k], dq_local[i]);
            }
        }
    } // q_base loop

    __syncthreads();

    // Float4 vectorized dK/dV write (de-padded, per-row)
    {
        const int stride4 = HeadDim / 4;
        for (int idx = threadIdx.x; idx < tile_size * stride4; idx += blockDim.x) {
            int r  = idx / stride4;
            int d4 = idx % stride4;
            int g_row = tile_start + r;
            int base  = r * HD_PAD + d4 * 4;
            float4 dk4 = make_float4(dKs[base], dKs[base+1], dKs[base+2], dKs[base+3]);
            float4 dv4 = make_float4(dVs[base], dVs[base+1], dVs[base+2], dVs[base+3]);
            reinterpret_cast<float4*>(dK_bh + g_row * HeadDim)[d4] = dk4;
            reinterpret_cast<float4*>(dV_bh + g_row * HeadDim)[d4] = dv4;
        }
    }
}

// ============================================================================
// Exp3: Combination — Fused A+B (smem atomics) + Float4 loads + HD_PAD
// ============================================================================
//
// Combines Exp1 and Exp2:
//   - HD_PAD smem stride on all tiles (bank-conflict free)
//   - float4 vectorized K/V loads and dK/dV writes (4x fewer transactions)
//   - Fused Phase A+B via smem atomics (no ds_buf/p_buf/Q_buf/dO_buf)
//   - One __syncthreads per q_base iteration only
//
// smem = 4 * BlockN * HD_PAD * sizeof(float)

template<int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp3(MemEfficientBwdParams params)
{
    constexpr int BlockN  = (HeadDim < 64) ? 32 : (2048 / HeadDim);
    constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD  = HeadDim + 1;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    extern __shared__ float smem[];
    float* Ks  = smem;
    float* Vs  = Ks  + BlockN * HD_PAD;
    float* dKs = Vs  + BlockN * HD_PAD;
    float* dVs = dKs + BlockN * HD_PAD;

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;

    const float* Q_bh   = params.Q   + bh_off;
    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* dO_bh  = params.dO  + bh_off;
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Float4 load K/V into padded smem; scalar-zero dKs/dVs (incl. padding slot)
    {
        const int stride4 = HeadDim / 4;
        for (int idx = threadIdx.x; idx < BlockN * stride4; idx += blockDim.x) {
            int r    = idx / stride4;
            int d4   = idx % stride4;
            int g_row = tile_start + r;
            float4 kv4 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 vv4 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (g_row < params.T) {
                kv4 = reinterpret_cast<const float4*>(K_bh + g_row * HeadDim)[d4];
                vv4 = reinterpret_cast<const float4*>(V_bh + g_row * HeadDim)[d4];
            }
            int base = r * HD_PAD + d4 * 4;
            Ks[base]   = kv4.x; Ks[base+1] = kv4.y;
            Ks[base+2] = kv4.z; Ks[base+3] = kv4.w;
            Vs[base]   = vv4.x; Vs[base+1] = vv4.y;
            Vs[base+2] = vv4.z; Vs[base+3] = vv4.w;
        }
        for (int idx = threadIdx.x; idx < BlockN * HD_PAD; idx += blockDim.x) {
            dKs[idx] = 0.0f;
            dVs[idx] = 0.0f;
        }
    }
    __syncthreads();

    const int q_start = Causal ? tile_start : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
        const int qi    = q_base + warp_id;
        const bool valid = (qi < params.T);

        float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
        float L_qi = 0.0f, D_qi = 0.0f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            int k = lane_id + i * BWD_WARP_SZ;
            float qv = 0.0f, dov = 0.0f;
            if (valid && k < HeadDim) {
                qv  = Q_bh [qi * HeadDim + k];
                dov = dO_bh[qi * HeadDim + k];
            }
            q_local[i]  = qv;
            do_local[i] = dov;
            dq_local[i] = 0.0f;
        }
        if (valid) { L_qi = LSE_bh[qi]; D_qi = D_bh[qi]; }

        // Fused Phase A+B with float4-loaded K/V (padded access)
        for (int j = 0; j < tile_size; ++j) {

            float dot_qk = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) dot_qk += q_local[i] * Ks[j * HD_PAD + k];
            }
            const float s = bwd_warp_sum(dot_qk) * params.scale;

            float p;
            if (!valid || (Causal && (tile_start + j) > qi))
                p = 0.0f;
            else
                p = exp2f(BWD_LOG2E * (s - L_qi));

            float dot_dov = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) dot_dov += do_local[i] * Vs[j * HD_PAD + k];
            }
            const float ds = p * (bwd_warp_sum(dot_dov) - D_qi);

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dq_local[i] += ds * params.scale * Ks[j * HD_PAD + k];
                    atomicAdd(&dKs[j * HD_PAD + k], ds * params.scale * q_local[i]);
                    atomicAdd(&dVs[j * HD_PAD + k], p  *                do_local[i]);
                }
            }
        } // j loop

        __syncthreads();  // smem atomic visibility barrier

        if (valid) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    atomicAdd(&dQ_bh[qi * HeadDim + k], dq_local[i]);
            }
        }
    } // q_base loop

    __syncthreads();

    // Float4 vectorized dK/dV write (de-padded, per-row)
    {
        const int stride4 = HeadDim / 4;
        for (int idx = threadIdx.x; idx < tile_size * stride4; idx += blockDim.x) {
            int r   = idx / stride4;
            int d4  = idx % stride4;
            int g_row = tile_start + r;
            int base  = r * HD_PAD + d4 * 4;
            float4 dk4 = make_float4(dKs[base], dKs[base+1], dKs[base+2], dKs[base+3]);
            float4 dv4 = make_float4(dVs[base], dVs[base+1], dVs[base+2], dVs[base+3]);
            reinterpret_cast<float4*>(dK_bh + g_row * HeadDim)[d4] = dk4;
            reinterpret_cast<float4*>(dV_bh + g_row * HeadDim)[d4] = dv4;
        }
    }
}

// ============================================================================
// Exp4: Merged QK+dOV i-loop + compile-time j-loop unrolling (builds on Exp2)
// ============================================================================
//
// Root cause of baseline ceiling (97% compute, still 10k below PyTorch):
//   Phase A runs TWO separate i-loops per j:
//     loop1: dot_qk  += q  * Ks[j]   → reads Ks[j]
//     warp_sum(dot_qk)
//     loop2: dot_dov += dO * Vs[j]   → reads Vs[j]   (second smem pass!)
//     warp_sum(dot_dov)
//   64 warp_sum calls per q_base (BlockN=32 × 2) = 320 shfl_xor per warp.
//   The two dot products are INDEPENDENT — no reason to compute them serially.
//
// Changes vs Exp2:
//   1. Merge the two i-loops into one: compute dot_qk AND dot_dov in the
//      same pass, reading Ks[j] and Vs[j] together (half the smem traffic).
//   2. Issue both warp_sum calls back-to-back after the merged loop.
//      No data dependency between them → PTX scheduler pipelines the shfl sequences.
//   3. j-loop bound changed from runtime tile_size → compile-time BlockN.
//      Out-of-bounds j: p=0, ds=0, ds_buf/p_buf store 0 (Phase B sums out safely).
//      This makes the loop body fully statically sized → #pragma unroll 4 works.
//   4. #pragma unroll 4 on j-loop: 4 independent j-groups visible simultaneously
//      to the instruction scheduler, hiding shuffle latency across groups.
//
// smem layout: identical to Exp2 (HD_PAD + full staging buffers)
// Dispatch: LAUNCH_MEM_BWD_EXP4 (same smem formula as EXP2)

template<int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp4(MemEfficientBwdParams params)
{
    constexpr int BlockN  = (HeadDim < 64) ? 32 : (2048 / HeadDim);
    constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD  = HeadDim + 1;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    extern __shared__ float smem[];
    float* Ks     = smem;
    float* Vs     = Ks     + BlockN  * HD_PAD;
    float* dKs    = Vs     + BlockN  * HD_PAD;
    float* dVs    = dKs    + BlockN  * HD_PAD;
    float* ds_buf = dVs    + BlockN  * HD_PAD;
    float* p_buf  = ds_buf + BWD_BLOCK_M * BlockN;
    float* Q_buf  = p_buf  + BWD_BLOCK_M * BlockN;
    float* dO_buf = Q_buf  + BWD_BLOCK_M * HD_PAD;

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;

    const float* Q_bh   = params.Q   + bh_off;
    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* dO_bh  = params.dO  + bh_off;
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Float4 load K/V into padded smem (same as Exp2)
    {
        const int stride4 = HeadDim / 4;
        for (int idx = threadIdx.x; idx < BlockN * stride4; idx += blockDim.x) {
            int r    = idx / stride4;
            int d4   = idx % stride4;
            int g_row = tile_start + r;
            float4 kv4 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 vv4 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (g_row < params.T) {
                kv4 = reinterpret_cast<const float4*>(K_bh + g_row * HeadDim)[d4];
                vv4 = reinterpret_cast<const float4*>(V_bh + g_row * HeadDim)[d4];
            }
            int base = r * HD_PAD + d4 * 4;
            Ks[base]   = kv4.x; Ks[base+1] = kv4.y;
            Ks[base+2] = kv4.z; Ks[base+3] = kv4.w;
            Vs[base]   = vv4.x; Vs[base+1] = vv4.y;
            Vs[base+2] = vv4.z; Vs[base+3] = vv4.w;
        }
        for (int idx = threadIdx.x; idx < BlockN * HD_PAD; idx += blockDim.x) {
            dKs[idx] = 0.0f;
            dVs[idx] = 0.0f;
        }
    }
    __syncthreads();

    const int q_start = Causal ? tile_start : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
        const int qi    = q_base + warp_id;
        const bool valid = (qi < params.T);

        float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
        float L_qi = 0.0f, D_qi = 0.0f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            int k = lane_id + i * BWD_WARP_SZ;
            float qv = 0.0f, dov = 0.0f;
            if (valid && k < HeadDim) {
                qv  = Q_bh [qi * HeadDim + k];
                dov = dO_bh[qi * HeadDim + k];
            }
            q_local[i]  = qv;
            do_local[i] = dov;
            dq_local[i] = 0.0f;
            if (k < HeadDim) {
                Q_buf [warp_id * HD_PAD + k] = qv;
                dO_buf[warp_id * HD_PAD + k] = dov;
            }
        }
        if (valid) { L_qi = LSE_bh[qi]; D_qi = D_bh[qi]; }

        // Phase A: iterate to compile-time BlockN so #pragma unroll 4 works.
        // Out-of-bounds j (j >= tile_size): p=0, ds=0 → zero contribution to dq/dK/dV.
        // Both dot products merged into ONE i-loop → single smem pass per j.
        // Both warp_sum calls issued back-to-back (no dependency) → scheduler pipelines them.
        #pragma unroll 4
        for (int j = 0; j < BlockN; ++j) {
            const bool j_valid = (j < tile_size);

            // Merged i-loop: compute dot_qk AND dot_dov in one smem pass
            float dot_qk = 0.0f, dot_dov = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dot_qk  += q_local[i]  * Ks[j * HD_PAD + k];
                    dot_dov += do_local[i] * Vs[j * HD_PAD + k];
                }
            }

            // Back-to-back warp_sum: no dependency between them, PTX scheduler
            // pipelines the two shfl_xor sequences across j groups (unrolled × 4)
            const float s   = bwd_warp_sum(dot_qk) * params.scale;
            const float dpv = bwd_warp_sum(dot_dov);

            float p = 0.0f;
            if (j_valid && valid && !(Causal && (tile_start + j) > qi))
                p = exp2f(BWD_LOG2E * (s - L_qi));

            const float ds = p * (dpv - D_qi);

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    dq_local[i] += ds * params.scale * Ks[j * HD_PAD + k];
            }

            // Always store (ds=0/p=0 for invalid j → Phase B accumulates 0 safely)
            if (lane_id == 0) {
                ds_buf[warp_id * BlockN + j] = ds;
                p_buf [warp_id * BlockN + j] = p;
            }
        } // j loop (unrolled × 4)

        __syncthreads();

        // Phase B: fully unrolled (BWD_BLOCK_M compile-time = 8), iterate to BlockN
        for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
            const int j = idx / HeadDim;
            const int k = idx % HeadDim;
            float dk_acc = 0.0f, dv_acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < BWD_BLOCK_M; ++w) {
                dk_acc += ds_buf[w * BlockN + j] * Q_buf [w * HD_PAD + k];
                dv_acc += p_buf [w * BlockN + j] * dO_buf[w * HD_PAD + k];
            }
            dKs[j * HD_PAD + k] += dk_acc * params.scale;
            dVs[j * HD_PAD + k] += dv_acc;
        }

        __syncthreads();

        if (valid) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    atomicAdd(&dQ_bh[qi * HeadDim + k], dq_local[i]);
            }
        }
    } // q_base loop

    __syncthreads();

    // Float4 dK/dV write (de-padded, same as Exp2)
    {
        const int stride4 = HeadDim / 4;
        for (int idx = threadIdx.x; idx < tile_size * stride4; idx += blockDim.x) {
            int r   = idx / stride4;
            int d4  = idx % stride4;
            int g_row = tile_start + r;
            int base  = r * HD_PAD + d4 * 4;
            float4 dk4 = make_float4(dKs[base], dKs[base+1], dKs[base+2], dKs[base+3]);
            float4 dv4 = make_float4(dVs[base], dVs[base+1], dVs[base+2], dVs[base+3]);
            reinterpret_cast<float4*>(dK_bh + g_row * HeadDim)[d4] = dk4;
            reinterpret_cast<float4*>(dV_bh + g_row * HeadDim)[d4] = dv4;
        }
    }
}

// ============================================================================
// Exp5: Halved BlockN → 100% SM occupancy + Fused A+B (no Phase B) + Float4
// ============================================================================
//
// Profiling diagnosis from exp2 Nsight data:
//   - 39.5 KB smem/block → 2 blocks/SM (102.4 KB SM limit) → 33% occupancy
//   - Phase B reads Q_buf/dO_buf: 16 smem accesses per output element →
//     MIO pipeline 99% saturated, 68% of all cycles are smem stalls
//   - L1/TEX 99% saturated, SM only 32% busy, 9% of FP32 peak achieved
//   - Issue slot utilization: 0.50/cycle (scheduler starved)
//
// Changes:
//   1. BlockN halved: (HeadDim<64)?16:(1024/HeadDim) instead of 2048/HeadDim
//      hd=64: BlockN 32→16, smem 4*16*65*4 = 16.25 KB → 6 blocks/SM → 100% occupancy
//      hd=128: BlockN 16→8, smem 4*8*129*4 = 16.13 KB → 6 blocks/SM → 100% occupancy
//      Effect: scheduler has 3× more warps to hide smem latency
//   2. Fused Phase A+B via smem atomics (no ds_buf, p_buf, Q_buf, dO_buf)
//      Eliminates Phase B entirely → no 16 smem reads per output element
//   3. Float4 K/V tile loads + HD_PAD (same as exp2/exp3)
//   4. Compile-time BlockN j-loop + #pragma unroll 4 (from exp4)
//
// smem = 4 * BlockN * (HeadDim+1) * sizeof(float) ≈ 16 KB

template<int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp5(MemEfficientBwdParams params)
{
    // Half the BlockN of baseline — key to fitting 6 blocks/SM on RTX 3060
    constexpr int BlockN  = (HeadDim < 64) ? 16 : (1024 / HeadDim);
    constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD  = HeadDim + 1;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    // smem: ONLY K, V, dK, dV tiles (no staging buffers → Phase B eliminated)
    extern __shared__ float smem[];
    float* Ks  = smem;
    float* Vs  = Ks  + BlockN * HD_PAD;
    float* dKs = Vs  + BlockN * HD_PAD;
    float* dVs = dKs + BlockN * HD_PAD;

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;

    const float* Q_bh   = params.Q   + bh_off;
    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* dO_bh  = params.dO  + bh_off;
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Float4 load K/V tile into padded smem; zero dKs/dVs
    {
        const int stride4 = HeadDim / 4;
        for (int idx = threadIdx.x; idx < BlockN * stride4; idx += blockDim.x) {
            int r    = idx / stride4;
            int d4   = idx % stride4;
            int g_row = tile_start + r;
            float4 kv4 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 vv4 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (g_row < params.T) {
                kv4 = reinterpret_cast<const float4*>(K_bh + g_row * HeadDim)[d4];
                vv4 = reinterpret_cast<const float4*>(V_bh + g_row * HeadDim)[d4];
            }
            int base = r * HD_PAD + d4 * 4;
            Ks[base]   = kv4.x; Ks[base+1] = kv4.y;
            Ks[base+2] = kv4.z; Ks[base+3] = kv4.w;
            Vs[base]   = vv4.x; Vs[base+1] = vv4.y;
            Vs[base+2] = vv4.z; Vs[base+3] = vv4.w;
        }
        for (int idx = threadIdx.x; idx < BlockN * HD_PAD; idx += blockDim.x) {
            dKs[idx] = 0.0f;
            dVs[idx] = 0.0f;
        }
    }
    __syncthreads();

    const int q_start = Causal ? tile_start : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
        const int qi    = q_base + warp_id;
        const bool valid = (qi < params.T);

        float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
        float L_qi = 0.0f, D_qi = 0.0f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            int k = lane_id + i * BWD_WARP_SZ;
            float qv = 0.0f, dov = 0.0f;
            if (valid && k < HeadDim) {
                qv  = Q_bh [qi * HeadDim + k];
                dov = dO_bh[qi * HeadDim + k];
            }
            q_local[i]  = qv;
            do_local[i] = dov;
            dq_local[i] = 0.0f;
        }
        if (valid) { L_qi = LSE_bh[qi]; D_qi = D_bh[qi]; }

        // Fused Phase A+B: merged QK+dOV i-loop (exp4) + smem atomics (exp3)
        // compile-time BlockN j-bound → #pragma unroll 4 works
        // q_local/do_local in registers → no Q_buf/dO_buf smem writes/reads needed
        #pragma unroll 4
        for (int j = 0; j < BlockN; ++j) {
            const bool j_valid = (j < tile_size);

            // Merged i-loop: read Ks[j] and Vs[j] in one smem pass
            float dot_qk = 0.0f, dot_dov = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dot_qk  += q_local[i]  * Ks[j * HD_PAD + k];
                    dot_dov += do_local[i] * Vs[j * HD_PAD + k];
                }
            }

            // Back-to-back warp_sums (independent → PTX pipelines them)
            const float s   = bwd_warp_sum(dot_qk) * params.scale;
            const float dpv = bwd_warp_sum(dot_dov);

            float p = 0.0f;
            if (j_valid && valid && !(Causal && (tile_start + j) > qi))
                p = exp2f(BWD_LOG2E * (s - L_qi));

            const float ds = p * (dpv - D_qi);

            // Fused: use q_local/do_local from registers → atomicAdd into dKs/dVs
            // No Q_buf/dO_buf smem needed. 8-way contention across warps but
            // 3× more warps in flight (100% occupancy) hides the serialization.
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dq_local[i] += ds * params.scale * Ks[j * HD_PAD + k];
                    atomicAdd(&dKs[j * HD_PAD + k], ds * params.scale * q_local[i]);
                    atomicAdd(&dVs[j * HD_PAD + k], p  *                do_local[i]);
                }
            }
        } // j-loop

        __syncthreads();  // atomics to dKs/dVs visible before next q_base

        if (valid) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    atomicAdd(&dQ_bh[qi * HeadDim + k], dq_local[i]);
            }
        }
    } // q_base loop

    __syncthreads();

    // Float4 dK/dV write (de-padded)
    {
        const int stride4 = HeadDim / 4;
        for (int idx = threadIdx.x; idx < tile_size * stride4; idx += blockDim.x) {
            int r   = idx / stride4;
            int d4  = idx % stride4;
            int g_row = tile_start + r;
            int base  = r * HD_PAD + d4 * 4;
            float4 dk4 = make_float4(dKs[base], dKs[base+1], dKs[base+2], dKs[base+3]);
            float4 dv4 = make_float4(dVs[base], dVs[base+1], dVs[base+2], dVs[base+3]);
            reinterpret_cast<float4*>(dK_bh + g_row * HeadDim)[d4] = dk4;
            reinterpret_cast<float4*>(dV_bh + g_row * HeadDim)[d4] = dv4;
        }
    }
}

// ============================================================================
// Exp6: Global atomics for dK/dV — eliminate dKs/dVs smem, remove MIO stalls
// ============================================================================
//
// Diagnosis from exp5: smem atomics to dKs/dVs still saturate MIO pipeline.
// Halved BlockN (100% occupancy) increased warps proportionally but also
// increased smem atomic load → net effect ≈ 0.
//
// Key insight: each KV-tile block owns UNIQUE rows [tile_start, tile_start+BlockN).
// → No inter-block contention for dK/dV whatsoever.
// → intra-block warp accumulation can go DIRECTLY to global dK_bh/dV_bh via
//   atomicAdd. Moves serialization from MIO pipeline → L2 subsystem (different
//   pipeline, higher throughput on Ampere, doesn't compete with smem reads).
//
// Zeroing strategy: each block zeros its own slice of dK_bh/dV_bh at startup
// (cooperative, one write per thread, __syncthreads before any atomicAdd).
// No external pre-zeroing required. Cost = BlockN*HeadDim / 256 writes/thread.
//
// smem: ONLY K and V tiles (2 * BlockN * HD_PAD ≈ 8.32 KB for hd=64)
//   → still 6 blocks/SM (warp-limited, not smem-limited)
//   → ONLY remaining smem accesses: Ks[j*HD_PAD+k] and Vs[j*HD_PAD+k] reads
//     (pure reads, not atomics → MIO handles at full throughput)
//
// Changes vs exp5:
//   1. dKs, dVs smem arrays REMOVED (saves 2*BlockN*HD_PAD = 8.32 KB smem)
//   2. startup: each block zeros its dK_bh/dV_bh rows (float4 for speed)
//   3. atomicAdd targets: global dK_bh/dV_bh instead of smem dKs/dVs
//   4. final dK/dV write loop REMOVED (no smem-to-global copy needed)
//   5. all other changes from exp5 preserved (halved BlockN, fused loop, float4)

template<int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp6(MemEfficientBwdParams params)
{
    constexpr int BlockN  = (HeadDim < 64) ? 16 : (1024 / HeadDim);
    constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD  = HeadDim + 1;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    // smem: ONLY K and V tiles — no dKs/dVs
    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = Ks + BlockN * HD_PAD;

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;

    const float* Q_bh   = params.Q   + bh_off;
    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* dO_bh  = params.dO  + bh_off;
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Step 0: zero this block's dK/dV slice in global memory (float4, coalesced)
    // Safe: this block exclusively owns rows [tile_start, tile_start+tile_size).
    // __syncthreads() after ensures zeros visible before any atomicAdd below.
    {
        const int stride4 = HeadDim / 4;
        for (int idx = threadIdx.x; idx < tile_size * stride4; idx += blockDim.x) {
            int r   = idx / stride4;
            int d4  = idx % stride4;
            int g_row = tile_start + r;
            reinterpret_cast<float4*>(dK_bh + g_row * HeadDim)[d4] =
                make_float4(0.f, 0.f, 0.f, 0.f);
            reinterpret_cast<float4*>(dV_bh + g_row * HeadDim)[d4] =
                make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    // Float4 load K/V tile into padded smem
    {
        const int stride4 = HeadDim / 4;
        for (int idx = threadIdx.x; idx < BlockN * stride4; idx += blockDim.x) {
            int r    = idx / stride4;
            int d4   = idx % stride4;
            int g_row = tile_start + r;
            float4 kv4 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 vv4 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (g_row < params.T) {
                kv4 = reinterpret_cast<const float4*>(K_bh + g_row * HeadDim)[d4];
                vv4 = reinterpret_cast<const float4*>(V_bh + g_row * HeadDim)[d4];
            }
            int base = r * HD_PAD + d4 * 4;
            Ks[base]   = kv4.x; Ks[base+1] = kv4.y;
            Ks[base+2] = kv4.z; Ks[base+3] = kv4.w;
            Vs[base]   = vv4.x; Vs[base+1] = vv4.y;
            Vs[base+2] = vv4.z; Vs[base+3] = vv4.w;
        }
    }

    // Barrier: K/V loaded, dK/dV zeros written to global — safe to proceed
    __syncthreads();

    const int q_start = Causal ? tile_start : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
        const int qi    = q_base + warp_id;
        const bool valid = (qi < params.T);

        float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
        float L_qi = 0.0f, D_qi = 0.0f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            int k = lane_id + i * BWD_WARP_SZ;
            float qv = 0.0f, dov = 0.0f;
            if (valid && k < HeadDim) {
                qv  = Q_bh [qi * HeadDim + k];
                dov = dO_bh[qi * HeadDim + k];
            }
            q_local[i]  = qv;
            do_local[i] = dov;
            dq_local[i] = 0.0f;
        }
        if (valid) { L_qi = LSE_bh[qi]; D_qi = D_bh[qi]; }

        // Fused Phase A+B: merged i-loop, compile-time j-loop, global atomics for dK/dV.
        // Only smem reads here (Ks, Vs) — no smem atomics → MIO stalls eliminated.
        #pragma unroll 4
        for (int j = 0; j < BlockN; ++j) {
            const bool j_valid = (j < tile_size);

            float dot_qk = 0.0f, dot_dov = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dot_qk  += q_local[i]  * Ks[j * HD_PAD + k];
                    dot_dov += do_local[i] * Vs[j * HD_PAD + k];
                }
            }

            const float s   = bwd_warp_sum(dot_qk) * params.scale;
            const float dpv = bwd_warp_sum(dot_dov);

            float p = 0.0f;
            if (j_valid && valid && !(Causal && (tile_start + j) > qi))
                p = exp2f(BWD_LOG2E * (s - L_qi));

            const float ds = p * (dpv - D_qi);

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dq_local[i] += ds * params.scale * Ks[j * HD_PAD + k];
                    // Global atomicAdd — moves to L2 pipeline, frees MIO for smem reads
                    atomicAdd(&dK_bh[(tile_start + j) * HeadDim + k],
                              ds * params.scale * q_local[i]);
                    atomicAdd(&dV_bh[(tile_start + j) * HeadDim + k],
                              p  *                do_local[i]);
                }
            }
        }

        // No __syncthreads needed here: global atomics are independent per-warp,
        // and dq_local is per-warp register. Next q_base reads fresh Q[qi+BWD_BLOCK_M].
        if (valid) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    atomicAdd(&dQ_bh[qi * HeadDim + k], dq_local[i]);
            }
        }
    } // q_base loop
    // No final dK/dV write needed — already in global memory via atomicAdd
}

// ----------------------------------------------------------------------------
// Exp7: Exp6 + bank-conflict-free K/V smem loading (scalar, not float4)
// Root cause of exp6's MIO throttle: float4 smem stores in K/V init use stride
// d4*4 within a row, so threads t and t+8 hit the same banks (stride 32).
// Fix: linear mapping idx→(r,k) so consecutive threads write consecutive banks.
// Everything else identical to exp6 (global atomics for dK/dV, BlockN=halved,
// HD_PAD, merged i-loop, unroll 4).
// Nsight exp6: Est. Speedup 45.88% from eliminating these bank conflicts.
// ----------------------------------------------------------------------------
template<int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp7(MemEfficientBwdParams params)
{
    constexpr int BlockN  = (HeadDim < 64) ? 16 : (1024 / HeadDim);
    constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD  = HeadDim + 1;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    // smem: ONLY K and V tiles — same as exp6
    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = Ks + BlockN * HD_PAD;

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;

    const float* Q_bh   = params.Q   + bh_off;
    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* dO_bh  = params.dO  + bh_off;
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Step 0: zero this block's dK/dV rows in global memory (scalar, coalesced)
    // Scalar avoids any float4 alignment edge cases; still fully coalesced.
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim;
        const int k = idx % HeadDim;
        dK_bh[(tile_start + r) * HeadDim + k] = 0.f;
        dV_bh[(tile_start + r) * HeadDim + k] = 0.f;
    }

    // Bank-conflict-free K/V smem load:
    // Thread idx → row r = idx/HeadDim, col k = idx%HeadDim.
    // Same-warp threads differ in k (consecutive) → consecutive smem banks → no conflict.
    // (float4 had stride d4*4 within row → threads t and t+8 collided on the same banks.)
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        const int r     = idx / HeadDim;
        const int k     = idx % HeadDim;
        const int g_row = tile_start + r;
        Ks[r * HD_PAD + k] = (g_row < params.T) ? K_bh[g_row * HeadDim + k] : 0.f;
        Vs[r * HD_PAD + k] = (g_row < params.T) ? V_bh[g_row * HeadDim + k] : 0.f;
    }

    // Barrier: K/V loaded, dK/dV zeroed — safe to proceed
    __syncthreads();

    const int q_start = Causal ? tile_start : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
        const int qi     = q_base + warp_id;
        const bool valid = (qi < params.T);

        float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
        float L_qi = 0.0f, D_qi = 0.0f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            const int k = lane_id + i * BWD_WARP_SZ;
            float qv = 0.0f, dov = 0.0f;
            if (valid && k < HeadDim) {
                qv  = Q_bh [qi * HeadDim + k];
                dov = dO_bh[qi * HeadDim + k];
            }
            q_local[i]  = qv;
            do_local[i] = dov;
            dq_local[i] = 0.0f;
        }
        if (valid) { L_qi = LSE_bh[qi]; D_qi = D_bh[qi]; }

        // Fused Phase A+B: same as exp6 — merged i-loop, global atomics for dK/dV.
        // Smem reads (Ks/Vs) are bank-conflict-free here since k varies within a warp.
        #pragma unroll 4
        for (int j = 0; j < BlockN; ++j) {
            const bool j_valid = (j < tile_size);

            float dot_qk = 0.0f, dot_dov = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dot_qk  += q_local[i]  * Ks[j * HD_PAD + k];
                    dot_dov += do_local[i] * Vs[j * HD_PAD + k];
                }
            }

            const float s   = bwd_warp_sum(dot_qk) * params.scale;
            const float dpv = bwd_warp_sum(dot_dov);

            float p = 0.0f;
            if (j_valid && valid && !(Causal && (tile_start + j) > qi))
                p = exp2f(BWD_LOG2E * (s - L_qi));

            const float ds = p * (dpv - D_qi);

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dq_local[i] += ds * params.scale * Ks[j * HD_PAD + k];
                    atomicAdd(&dK_bh[(tile_start + j) * HeadDim + k],
                              ds * params.scale * q_local[i]);
                    atomicAdd(&dV_bh[(tile_start + j) * HeadDim + k],
                              p  *                do_local[i]);
                }
            }
        }

        if (valid) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    atomicAdd(&dQ_bh[qi * HeadDim + k], dq_local[i]);
            }
        }
    } // q_base loop
}

// ----------------------------------------------------------------------------
// Exp8: WMMA tensor-core backward
// Uses fp16 tensor cores for dK and dV accumulation.
// BM_WMMA=16 (2 q-rows per warp), BlockN=16.
// Phase A: scalar QK^T→ds/p written to fp16 smem for WMMA.
// Phase B: warps 0-3 accumulate dK, warps 4-7 accumulate dV via wmma::mma_sync.
//   dK/dV fragments persist in registers across the entire q_base loop.
//   Written to global ONCE at end — no atomics for dK/dV.
// dQ: scalar global atomicAdd (cross-block dep, unavoidable).
// Only valid when HeadDim is a multiple of 16 (WMMA tile requirement).
// ----------------------------------------------------------------------------
template<int HeadDim, bool Causal>
__global__ __launch_bounds__(256, 3)
void mem_efficient_bwd_unified_kernel_exp8(MemEfficientBwdParams params)
{
    using namespace nvcuda;
    constexpr int BlockN    = 16;
    constexpr int BM_WMMA   = 16;            // 8 warps × 2 q-rows
    constexpr int HD_CHUNKS = HeadDim / 16;  // 4 for HD=64
    constexpr int LocalN    = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD    = HeadDim + 1;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    // ── smem layout ──────────────────────────────────────────────────────────
    // fp32: Ks[BlockN][HD_PAD], Vs[BlockN][HD_PAD]   (loaded once, K/V tile)
    // fp16: Q_h[BM_WMMA][HeadDim], dO_h[BM_WMMA][HeadDim]  (per q_base)
    //       ds_T_h[BlockN][BM_WMMA], p_T_h[BlockN][BM_WMMA]  (ds/p transposed)
    // At kernel end: Ks/Vs reused as dK_st/dV_st [BlockN*HeadDim] linear.
    extern __shared__ float smem_f[];
    float* Ks      = smem_f;
    float* Vs      = Ks + BlockN * HD_PAD;
    __half* smem_h = reinterpret_cast<__half*>(Vs + BlockN * HD_PAD);
    __half* Q_h    = smem_h;
    __half* dO_h   = Q_h   + BM_WMMA * HeadDim;
    __half* ds_T_h = dO_h  + BM_WMMA * HeadDim;
    __half* p_T_h  = ds_T_h + BlockN * BM_WMMA;

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;

    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* Q_bh   = params.Q   + bh_off;   // fp32 global — Phase A dot products
    const float* dO_bh  = params.dO  + bh_off;   // fp32 global — Phase A dot products
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Step 0: zero this block's dK/dV rows
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dK_bh[(tile_start + r) * HeadDim + k] = 0.f;
        dV_bh[(tile_start + r) * HeadDim + k] = 0.f;
    }

    // Step 1: load K/V tile into fp32 smem (scalar Phase A dot products)
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        const int g = tile_start + r;
        Ks[r * HD_PAD + k] = (g < params.T) ? K_bh[g * HeadDim + k] : 0.f;
        Vs[r * HD_PAD + k] = (g < params.T) ? V_bh[g * HeadDim + k] : 0.f;
    }
    __syncthreads();

    // Persistent WMMA accumulator: warps 0-3 → dK, warps 4-7 → dV
    const int chunk = warp_id % HD_CHUNKS;  // HeadDim slice (0-3 for HD=64)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    const int q_start = Causal ? tile_start : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BM_WMMA) {
        // Sync ensures prior Phase B WMMA reads are complete before overwriting smem
        __syncthreads();

        // Load Q and dO to fp16 smem (all warps cooperate) — used by Phase B WMMA only
        for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            const int qi = q_base + r;
            const bool vq = (qi < params.T);
            Q_h [r * HeadDim + k] = vq ? __float2half(Q_bh [qi * HeadDim + k]) : __float2half(0.f);
            dO_h[r * HeadDim + k] = vq ? __float2half(dO_bh[qi * HeadDim + k]) : __float2half(0.f);
        }
        // Sync: Q_h/dO_h fully written before Phase B reads them
        __syncthreads();

        // ── Phase A: fp32 global reads (no fp16 precision loss in dot products) ──
        const int qi0 = q_base + warp_id * 2;
        const int qi1 = qi0 + 1;
        const bool v0 = (qi0 < params.T);
        const bool v1 = (qi1 < params.T);

        float q0[LocalN], q1[LocalN], do0[LocalN], do1[LocalN];
        float dq0[LocalN], dq1[LocalN];
        float L0 = 0.f, D0 = 0.f, L1 = 0.f, D1 = 0.f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            const int k = lane_id + i * BWD_WARP_SZ;
            // fp32 global reads — L2 cache hit (same lines brought in by Q_h/dO_h load above)
            q0[i]  = (k < HeadDim && v0) ? Q_bh [qi0 * HeadDim + k] : 0.f;
            q1[i]  = (k < HeadDim && v1) ? Q_bh [qi1 * HeadDim + k] : 0.f;
            do0[i] = (k < HeadDim && v0) ? dO_bh[qi0 * HeadDim + k] : 0.f;
            do1[i] = (k < HeadDim && v1) ? dO_bh[qi1 * HeadDim + k] : 0.f;
            dq0[i] = dq1[i] = 0.f;
        }
        if (v0) { L0 = LSE_bh[qi0]; D0 = D_bh[qi0]; }
        if (v1) { L1 = LSE_bh[qi1]; D1 = D_bh[qi1]; }

        #pragma unroll 4
        for (int j = 0; j < BlockN; ++j) {
            const bool j_valid = (j < tile_size);
            float dqk0 = 0.f, dov0 = 0.f, dqk1 = 0.f, dov1 = 0.f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dqk0 += q0[i] * Ks[j * HD_PAD + k];
                    dov0 += do0[i] * Vs[j * HD_PAD + k];
                    dqk1 += q1[i] * Ks[j * HD_PAD + k];
                    dov1 += do1[i] * Vs[j * HD_PAD + k];
                }
            }
            const float s0   = bwd_warp_sum(dqk0) * params.scale;
            const float dpv0 = bwd_warp_sum(dov0);
            const float s1   = bwd_warp_sum(dqk1) * params.scale;
            const float dpv1 = bwd_warp_sum(dov1);

            float p0 = 0.f, p1 = 0.f;
            if (j_valid && v0 && !(Causal && (tile_start + j) > qi0))
                p0 = exp2f(BWD_LOG2E * (s0 - L0));
            if (j_valid && v1 && !(Causal && (tile_start + j) > qi1))
                p1 = exp2f(BWD_LOG2E * (s1 - L1));

            const float ds0 = p0 * (dpv0 - D0);
            const float ds1 = p1 * (dpv1 - D1);

            // Write ds*scale (for dK) and p (for dV) to fp16 smem, transposed layout.
            // Clamp ds to fp16 safe range to prevent inf/NaN propagation through WMMA.
            if (lane_id == 0) {
                ds_T_h[j * BM_WMMA + warp_id * 2    ] = __float2half(fmaxf(-65000.f, fminf(65000.f, ds0 * params.scale)));
                ds_T_h[j * BM_WMMA + warp_id * 2 + 1] = __float2half(fmaxf(-65000.f, fminf(65000.f, ds1 * params.scale)));
                p_T_h [j * BM_WMMA + warp_id * 2    ] = __float2half(p0);
                p_T_h [j * BM_WMMA + warp_id * 2 + 1] = __float2half(p1);
            }

            // Accumulate dQ in fp32 registers (written to global below)
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dq0[i] += ds0 * params.scale * Ks[j * HD_PAD + k];
                    dq1[i] += ds1 * params.scale * Ks[j * HD_PAD + k];
                }
            }
        } // j loop

        // dQ global atomicAdd (cross-block dep — unavoidable)
        if (v0) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) atomicAdd(&dQ_bh[qi0 * HeadDim + k], dq0[i]);
            }
        }
        if (v1) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) atomicAdd(&dQ_bh[qi1 * HeadDim + k], dq1[i]);
            }
        }

        // ── Phase B: WMMA — accumulate dK (warps 0-3) or dV (warps 4-7) ────
        // Sync: all warps must finish writing ds_T_h / p_T_h before any warp reads them
        __syncthreads();
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            // A = ds^T (for dK) or p^T (for dV); B = Q chunk (for dK) or dO chunk (for dV)
            const __half* a_ptr = (warp_id < HD_CHUNKS) ? ds_T_h         : p_T_h;
            const __half* b_ptr = (warp_id < HD_CHUNKS) ? (Q_h  + chunk * 16)
                                                         : (dO_h + chunk * 16);
            wmma::load_matrix_sync(a_frag, a_ptr, BM_WMMA);  // [BlockN × BM_WMMA]
            wmma::load_matrix_sync(b_frag, b_ptr, HeadDim);  // [BM_WMMA × 16]
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        // (next iteration's opening __syncthreads() covers Phase B completion)
    } // q_base loop

    // ── Store WMMA fragments to global dK/dV (no atomics, block owns rows) ──
    __syncthreads();
    // Reuse Ks/Vs smem as linear [BlockN×HeadDim] store buffers
    float* dK_st = Ks;
    float* dV_st = Vs;
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        dK_st[idx] = 0.f;
        dV_st[idx] = 0.f;
    }
    __syncthreads();
    {
        float* out = (warp_id < HD_CHUNKS) ? dK_st : dV_st;
        // Store [BlockN×16] tile at columns [chunk*16 .. chunk*16+15], stride HeadDim
        wmma::store_matrix_sync(out + chunk * 16, acc_frag, HeadDim, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dK_bh[(tile_start + r) * HeadDim + k] = dK_st[r * HeadDim + k];
        dV_bh[(tile_start + r) * HeadDim + k] = dV_st[r * HeadDim + k];
    }
}

// ============================================================================
// Exp9: TF32 WMMA — fair comparison with PyTorch (same precision as cuBLAS)
//
// PyTorch uses TF32 tensor cores (fp32 range, 10-bit mantissa) for fp32 inputs.
// Exp8/9 used fp16 WMMA (smaller range ±65504) — unfair advantage.
// Exp10 uses wmma::precision::tf32 with [16,16,8] tiles:
//   - Inputs: fp32 (no __float2half, no clamp needed)
//   - k=8 per MMA → 2 calls per chunk to cover BM_WMMA=16
//   - Accumulator: float (same as before)
// All smem stays fp32 — ds_T, p_T, Q_sm, dO_sm.
// ============================================================================

template <int HeadDim, bool Causal>
__launch_bounds__(256, 3)
__global__ void mem_efficient_bwd_unified_kernel_exp9(MemEfficientBwdParams params)
{
    using namespace nvcuda;
    constexpr int BlockN    = 16;
    constexpr int BM_WMMA   = 16;
    constexpr int HD_CHUNKS = HeadDim / 16;
    constexpr int LocalN    = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD    = HeadDim + 1;  // fp32 smem: +1 float = 32-row bank cycle
    constexpr int BM_PAD    = BM_WMMA + 1;  // fp32 smem: +1 float = 32-row bank cycle
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    // smem layout (all fp32 — TF32 WMMA takes fp32 inputs directly):
    // Ks[BlockN][HD_PAD], Vs[BlockN][HD_PAD]
    // ds_T[BlockN][BM_PAD], p_T[BlockN][BM_PAD]
    // Q_sm[BM_WMMA][HD_PAD], dO_sm[BM_WMMA][HD_PAD]
    extern __shared__ float smem_f[];
    float* Ks    = smem_f;
    float* Vs    = Ks    + BlockN  * HD_PAD;
    float* ds_T  = Vs    + BlockN  * HD_PAD;
    float* p_T   = ds_T  + BlockN  * BM_PAD;
    float* Q_sm  = p_T   + BlockN  * BM_PAD;
    float* dO_sm = Q_sm  + BM_WMMA * HD_PAD;

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;

    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* Q_bh   = params.Q   + bh_off;
    const float* dO_bh  = params.dO  + bh_off;
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Step 0: zero dK/dV rows
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dK_bh[(tile_start + r) * HeadDim + k] = 0.f;
        dV_bh[(tile_start + r) * HeadDim + k] = 0.f;
    }

    // Step 1: load K/V tile into fp32 smem
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        const int g = tile_start + r;
        Ks[r * HD_PAD + k] = (g < params.T) ? K_bh[g * HeadDim + k] : 0.f;
        Vs[r * HD_PAD + k] = (g < params.T) ? V_bh[g * HeadDim + k] : 0.f;
    }
    __syncthreads();

    // Persistent TF32 WMMA accumulator (m=16, n=16, k=8; acc is float)
    const int chunk = warp_id % HD_CHUNKS;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    const int q_start = Causal ? max(0, params.k_offset + tile_start - params.q_offset) : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BM_WMMA) {
        // Sync (a): prior Phase B WMMA reads done; safe to overwrite Q_sm/dO_sm
        __syncthreads();

        // Load Q and dO into fp32 smem with bounds check (for Phase B WMMA)
        for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            const int qi = q_base + r;
            const bool vq = (qi < params.T);
            Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * HeadDim + k] : 0.f;
            dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * HeadDim + k] : 0.f;
        }
        // Sync (b): Q_sm/dO_sm fully written
        __syncthreads();

        // ── Phase A: fp32 global reads ────────────────────────────────────────
        const int qi0 = q_base + warp_id * 2;
        const int qi1 = qi0 + 1;
        const bool v0 = (qi0 < params.T);
        const bool v1 = (qi1 < params.T);

        float q0[LocalN], q1[LocalN], do0[LocalN], do1[LocalN];
        float dq0[LocalN], dq1[LocalN];
        float L0 = 0.f, D0 = 0.f, L1 = 0.f, D1 = 0.f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            const int k = lane_id + i * BWD_WARP_SZ;
            q0[i]  = (k < HeadDim && v0) ? Q_bh [qi0 * HeadDim + k] : 0.f;
            q1[i]  = (k < HeadDim && v1) ? Q_bh [qi1 * HeadDim + k] : 0.f;
            do0[i] = (k < HeadDim && v0) ? dO_bh[qi0 * HeadDim + k] : 0.f;
            do1[i] = (k < HeadDim && v1) ? dO_bh[qi1 * HeadDim + k] : 0.f;
            dq0[i] = dq1[i] = 0.f;
        }
        if (v0) { L0 = LSE_bh[qi0]; D0 = D_bh[qi0]; }
        if (v1) { L1 = LSE_bh[qi1]; D1 = D_bh[qi1]; }

        #pragma unroll 4
        for (int j = 0; j < BlockN; ++j) {
            const bool j_valid = (j < tile_size);
            float dqk0 = 0.f, dov0 = 0.f, dqk1 = 0.f, dov1 = 0.f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dqk0 += q0[i] * Ks[j * HD_PAD + k];
                    dov0 += do0[i] * Vs[j * HD_PAD + k];
                    dqk1 += q1[i] * Ks[j * HD_PAD + k];
                    dov1 += do1[i] * Vs[j * HD_PAD + k];
                }
            }
            const float s0   = bwd_warp_sum(dqk0) * params.scale;
            const float dpv0 = bwd_warp_sum(dov0);
            const float s1   = bwd_warp_sum(dqk1) * params.scale;
            const float dpv1 = bwd_warp_sum(dov1);

            float p0 = 0.f, p1 = 0.f;
            if (j_valid && v0 && !(Causal && (params.k_offset + tile_start + j) > (params.q_offset + qi0)))
                p0 = exp2f(BWD_LOG2E * (s0 - L0));
            if (j_valid && v1 && !(Causal && (params.k_offset + tile_start + j) > (params.q_offset + qi1)))
                p1 = exp2f(BWD_LOG2E * (s1 - L1));

            const float ds0 = p0 * (dpv0 - D0);
            const float ds1 = p1 * (dpv1 - D1);

            // Write fp32 ds*scale and p to smem (transposed) — no clamp needed
            if (lane_id == 0) {
                ds_T[j * BM_PAD + warp_id * 2    ] = ds0 * params.scale;
                ds_T[j * BM_PAD + warp_id * 2 + 1] = ds1 * params.scale;
                p_T [j * BM_PAD + warp_id * 2    ] = p0;
                p_T [j * BM_PAD + warp_id * 2 + 1] = p1;
            }

            // Accumulate dQ in fp32 registers
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dq0[i] += ds0 * params.scale * Ks[j * HD_PAD + k];
                    dq1[i] += ds1 * params.scale * Ks[j * HD_PAD + k];
                }
            }
        } // j loop

        // dQ global atomicAdd (cross-block dep — unavoidable)
        if (v0) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) atomicAdd(&dQ_bh[qi0 * HeadDim + k], dq0[i]);
            }
        }
        if (v1) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) atomicAdd(&dQ_bh[qi1 * HeadDim + k], dq1[i]);
            }
        }

        // ── Phase B: TF32 WMMA — 2×MMA per chunk (k=8+8 covers BM_WMMA=16) ──
        // Sync (c): ds_T/p_T written; Q_sm/dO_sm already ready from sync (b)
        __syncthreads();
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;

            const float* a_ptr = (warp_id < HD_CHUNKS) ? ds_T  : p_T;
            const float* b_ptr = ((warp_id < HD_CHUNKS) ? Q_sm  : dO_sm) + chunk * 16;

            // k-split 0: A[:,0:8], B[0:8,chunk*16:chunk*16+16]
            wmma::load_matrix_sync(a_frag, a_ptr,              BM_PAD);  // [16×8] smem
            wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);  // [8×16] smem
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

            // k-split 1: A[:,8:16], B[8:16,chunk*16:chunk*16+16]
            wmma::load_matrix_sync(a_frag, a_ptr + 8,          BM_PAD);  // [16×8], cols 8..15
            wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);  // [8×16], rows 8..15
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        // (next iteration's sync (a) ensures Phase B reads are done)
    } // q_base loop

    // ── Store TF32 WMMA fragments to global dK/dV (no atomics) ──────────────
    __syncthreads();
    float* dK_st = Ks;
    float* dV_st = Vs;
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        dK_st[idx] = 0.f;
        dV_st[idx] = 0.f;
    }
    __syncthreads();
    {
        float* out = (warp_id < HD_CHUNKS) ? dK_st : dV_st;
        wmma::store_matrix_sync(out + chunk * 16, acc_frag, HeadDim, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dK_bh[(tile_start + r) * HeadDim + k] = dK_st[r * HeadDim + k];
        dV_bh[(tile_start + r) * HeadDim + k] = dV_st[r * HeadDim + k];
    }
}

// ============================================================================
// Exp10: 32×32 tile TF32 WMMA backward (2× larger tiles than exp9)
//   - BlockN=32 (KV tile), BM=32 (Q tile) → halves iterations + syncs
//   - Phase A: 4 Q rows per warp (scalar score/P/ds/dQ)
//   - Phase B: paired WMMA acc (top/bot 16-row halves of 32-row output)
//   - smem ≈ 42KB → fits default 48KB, 2 blocks/SM on sm_86
// ============================================================================

template <int HeadDim, bool Causal>
__launch_bounds__(256, 2)
__global__ void mem_efficient_bwd_unified_kernel_exp10(MemEfficientBwdParams params)
{
    using namespace nvcuda;
    constexpr int BlockN    = 32;
    constexpr int BM        = 32;
    constexpr int HD_CHUNKS = HeadDim / 16;
    constexpr int LocalN    = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD    = HeadDim + 1;
    constexpr int BM_PAD    = BM + 1;
    constexpr int K_SPLITS  = BM / 8;  // 4
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    extern __shared__ float smem_f[];
    float* Ks    = smem_f;
    float* Vs    = Ks    + BlockN * HD_PAD;
    float* ds_T  = Vs    + BlockN * HD_PAD;
    float* p_T   = ds_T  + BlockN * BM_PAD;
    float* Q_sm  = p_T   + BlockN * BM_PAD;
    float* dO_sm = Q_sm  + BM     * HD_PAD;

    const int bh         = blockIdx.y;
    const int tile_start = blockIdx.x * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;

    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* Q_bh   = params.Q   + bh_off;
    const float* dO_bh  = params.dO  + bh_off;
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Zero dK/dV
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dK_bh[(tile_start + r) * HeadDim + k] = 0.f;
        dV_bh[(tile_start + r) * HeadDim + k] = 0.f;
    }

    // Load K/V tiles
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        const int g = tile_start + r;
        Ks[r * HD_PAD + k] = (g < params.T) ? K_bh[g * HeadDim + k] : 0.f;
        Vs[r * HD_PAD + k] = (g < params.T) ? V_bh[g * HeadDim + k] : 0.f;
    }
    __syncthreads();

    // WMMA accumulators: top/bot halves of [BlockN=32 × 16] output per chunk
    const int chunk = warp_id % HD_CHUNKS;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_top, acc_bot;
    wmma::fill_fragment(acc_top, 0.0f);
    wmma::fill_fragment(acc_bot, 0.0f);

    const int q_start = Causal ? max(0, params.k_offset + tile_start - params.q_offset) : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BM) {
        __syncthreads();

        // Load Q, dO into smem
        for (int idx = threadIdx.x; idx < BM * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            const int qi = q_base + r;
            const bool vq = (qi < params.T);
            Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * HeadDim + k] : 0.f;
            dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * HeadDim + k] : 0.f;
        }
        __syncthreads();

        // Phase A: 4 Q rows per warp (warp_id*4 .. warp_id*4+3)
        const int qi_base = q_base + warp_id * 4;
        float q_r[4][LocalN], do_r[4][LocalN], dq_r[4][LocalN];
        float L_r[4], D_r[4];
        bool v_r[4];

        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int qi = qi_base + w;
            v_r[w] = (qi < params.T);
            L_r[w] = v_r[w] ? LSE_bh[qi] : 0.f;
            D_r[w] = v_r[w] ? D_bh[qi]   : 0.f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                q_r[w][i]  = (k < HeadDim && v_r[w]) ? Q_bh [qi * HeadDim + k] : 0.f;
                do_r[w][i] = (k < HeadDim && v_r[w]) ? dO_bh[qi * HeadDim + k] : 0.f;
                dq_r[w][i] = 0.f;
            }
        }

        #pragma unroll 4
        for (int j = 0; j < BlockN; ++j) {
            const bool j_valid = (j < tile_size);

            float dqk[4], dov[4];
            #pragma unroll
            for (int w = 0; w < 4; ++w) { dqk[w] = 0.f; dov[w] = 0.f; }

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    const float kv = Ks[j * HD_PAD + k];
                    const float vv = Vs[j * HD_PAD + k];
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        dqk[w] += q_r[w][i] * kv;
                        dov[w] += do_r[w][i] * vv;
                    }
                }
            }

            float p[4], ds[4];
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                float s   = bwd_warp_sum(dqk[w]) * params.scale;
                float dpv = bwd_warp_sum(dov[w]);
                const int qi = qi_base + w;
                p[w] = 0.f;
                if (j_valid && v_r[w] && !(Causal && (params.k_offset + tile_start + j) > (params.q_offset + qi)))
                    p[w] = exp2f(BWD_LOG2E * (s - L_r[w]));
                ds[w] = p[w] * (dpv - D_r[w]);

                if (lane_id == 0) {
                    ds_T[j * BM_PAD + warp_id * 4 + w] = ds[w] * params.scale;
                    p_T [j * BM_PAD + warp_id * 4 + w] = p[w];
                }
            }

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    const float kv = Ks[j * HD_PAD + k];
                    #pragma unroll
                    for (int w = 0; w < 4; ++w)
                        dq_r[w][i] += ds[w] * params.scale * kv;
                }
            }
        }

        // dQ atomicAdd
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (v_r[w]) {
                const int qi = qi_base + w;
                #pragma unroll
                for (int i = 0; i < LocalN; ++i) {
                    const int k = lane_id + i * BWD_WARP_SZ;
                    if (k < HeadDim) atomicAdd(&dQ_bh[qi * HeadDim + k], dq_r[w][i]);
                }
            }
        }

        // Phase B: TF32 WMMA for dK/dV with paired top/bot accumulators
        __syncthreads();
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;

            const float* a_ptr = (warp_id < HD_CHUNKS) ? ds_T : p_T;
            const float* b_ptr = ((warp_id < HD_CHUNKS) ? Q_sm : dO_sm) + chunk * 16;

            #pragma unroll
            for (int ks = 0; ks < K_SPLITS; ++ks) {
                // Load B once (shared between top and bot)
                wmma::load_matrix_sync(b_frag, b_ptr + ks * 8 * HD_PAD, HD_PAD);
                // Top half: rows 0..15 of output
                wmma::load_matrix_sync(a_frag, a_ptr + ks * 8, BM_PAD);
                wmma::mma_sync(acc_top, a_frag, b_frag, acc_top);
                // Bot half: rows 16..31 of output
                wmma::load_matrix_sync(a_frag, a_ptr + 16 * BM_PAD + ks * 8, BM_PAD);
                wmma::mma_sync(acc_bot, a_frag, b_frag, acc_bot);
            }
        }
    }

    // Store WMMA fragments → smem → global
    __syncthreads();
    float* dK_st = Ks;
    float* dV_st = Vs;
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        dK_st[idx] = 0.f;
        dV_st[idx] = 0.f;
    }
    __syncthreads();
    {
        float* out = (warp_id < HD_CHUNKS) ? dK_st : dV_st;
        wmma::store_matrix_sync(out + chunk * 16,                     acc_top, HeadDim, wmma::mem_row_major);
        wmma::store_matrix_sync(out + 16 * HeadDim + chunk * 16, acc_bot, HeadDim, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dK_bh[(tile_start + r) * HeadDim + k] = dK_st[r * HeadDim + k];
        dV_bh[(tile_start + r) * HeadDim + k] = dV_st[r * HeadDim + k];
    }
}

// ============================================================================
// Public API (namespace cuda)
// ============================================================================

namespace cuda {

void compute_row_lse(
    const float* scores, float* lse,
    int64_t batch, int64_t T, bool is_causal)
{
    int64_t total_rows = batch * T;
    int threads = 32;
    while (threads < T && threads < 1024) threads <<= 1;
    size_t smem = threads * sizeof(float);
    compute_row_lse_kernel<<<(int)total_rows, threads, smem>>>(
        scores, lse, T, is_causal);
}

void mem_efficient_attn_forward(
    const float* query, const float* key, const float* value,
    float* output, float* lse,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal,
    float dropout_p, const float* dropout_mask)
{
    if (hd > MAX_HD) {
        printf("mem_efficient_attn_forward: hd=%d exceeds MAX_HD=%d\n",
               (int)hd, MAX_HD);
        return;
    }
    float scale = 1.0f / sqrtf(static_cast<float>(hd));
    int grid_y = (int)(B * nh);
    launch_fwd_tc_kernel(query, key, value, output, lse,
                      T, hd, scale, is_causal,
                      dropout_p, dropout_mask, grid_y);
}

void mem_efficient_attn_forward_tc(
    const float* query, const float* key, const float* value,
    float* output, float* lse,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal,
    float dropout_p, const float* dropout_mask,
    int q_offset, int k_offset)
{
    if (hd > MAX_HD) {
        printf("mem_efficient_attn_forward_tc: hd=%d exceeds MAX_HD=%d\n",
               (int)hd, MAX_HD);
        return;
    }
    float scale = 1.0f / sqrtf(static_cast<float>(hd));
    int grid_y = (int)(B * nh);
    launch_fwd_tc_kernel(query, key, value, output, lse,
                         T, hd, scale, is_causal,
                         dropout_p, dropout_mask, grid_y,
                         q_offset, k_offset);
}

void mem_efficient_attn_backward(
    const float* query, const float* key, const float* value,
    const float* output, const float* grad_output, const float* lse,
    float* grad_query, float* grad_key, float* grad_value,
    float* D_buf,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal,
    int q_offset,
    int k_offset)
{
    float scale = 1.0f / sqrtf(static_cast<float>(hd));
    const int BH = (int)(B * nh);
    dim3 block_cfg(BWD_NUM_THREADS);

    auto get_block_n = [](int d) -> int {
        return (d < 64) ? 32 : (2048 / d);
    };
    int block_n = get_block_n((int)hd);

    dim3 grid_D((T + BWD_BLOCK_M - 1) / BWD_BLOCK_M, BH);
    int kv_tiles = ((int)T + block_n - 1) / block_n;
    dim3 grid_bwd(kv_tiles, BH);

    [[maybe_unused]] size_t shmem_bwd = (4ULL * block_n * hd
                      + 2ULL * BWD_BLOCK_M * block_n
                      + 2ULL * BWD_BLOCK_M * hd) * sizeof(float);

    MemEfficientBwdParams params;
    params.Q     = query;
    params.K     = key;
    params.V     = value;
    params.dO    = grad_output;
    params.LSE   = lse;
    params.D     = D_buf;
    params.dQ    = grad_query;
    params.dK    = grad_key;
    params.dV    = grad_value;
    params.T     = (int)T;
    params.scale = scale;
    params.is_causal = is_causal;
    params.q_offset  = q_offset;
    params.k_offset  = k_offset;

    // Original kernel (unchanged — baseline reference)
#define LAUNCH_MEM_BWD(HD) \
    do { \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel<HD, true><<<grid_bwd, block_cfg, shmem_bwd>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel<HD, false><<<grid_bwd, block_cfg, shmem_bwd>>>(params); \
        } \
    } while (0)

    // Exp3: Fused A+B + float4 + HD_PAD  (active — swap macro below to A/B to compare)
    // smem = 4 * BlockN * (HeadDim+1): no staging buffers, padded smem stride
#define LAUNCH_MEM_BWD_EXP3(HD) \
    do { \
        const size_t shmem_exp3 = 4ULL * block_n * ((HD) + 1) * sizeof(float); \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp3<HD, true> \
                <<<grid_bwd, block_cfg, shmem_exp3>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp3<HD, false> \
                <<<grid_bwd, block_cfg, shmem_exp3>>>(params); \
        } \
    } while (0)

    // Exp1: Fused A+B via smem atomics + HD_PAD  (no float4)
    // smem = 4 * BlockN * (HeadDim+1)
#define LAUNCH_MEM_BWD_EXP1(HD) \
    do { \
        const size_t shmem_exp1 = 4ULL * block_n * ((HD) + 1) * sizeof(float); \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp1<HD, true> \
                <<<grid_bwd, block_cfg, shmem_exp1>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp1<HD, false> \
                <<<grid_bwd, block_cfg, shmem_exp1>>>(params); \
        } \
    } while (0)

    // Exp2: Staged A+B + float4 + HD_PAD  (no smem atomics)
    // smem = 4*BlockN*HD_PAD + 2*BM*BlockN + 2*BM*HD_PAD
#define LAUNCH_MEM_BWD_EXP2(HD) \
    do { \
        const size_t shmem_exp2 = (4ULL * block_n * ((HD) + 1) \
                                 + 2ULL * BWD_BLOCK_M * block_n \
                                 + 2ULL * BWD_BLOCK_M * ((HD) + 1)) * sizeof(float); \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp2<HD, true> \
                <<<grid_bwd, block_cfg, shmem_exp2>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp2<HD, false> \
                <<<grid_bwd, block_cfg, shmem_exp2>>>(params); \
        } \
    } while (0)

    // Exp4: Merged QK+dOV i-loop + compile-time j-loop + #pragma unroll 4
    // smem: same formula as Exp2 (HD_PAD staging buffers)
#define LAUNCH_MEM_BWD_EXP4(HD) \
    do { \
        const size_t shmem_exp4 = (4ULL * block_n * ((HD) + 1) \
                                 + 2ULL * BWD_BLOCK_M * block_n \
                                 + 2ULL * BWD_BLOCK_M * ((HD) + 1)) * sizeof(float); \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp4<HD, true> \
                <<<grid_bwd, block_cfg, shmem_exp4>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp4<HD, false> \
                <<<grid_bwd, block_cfg, shmem_exp4>>>(params); \
        } \
    } while (0)

    // Exp5: Halved BlockN (100% SM occupancy) + Fused A+B + Float4 + merged loop
    // block_n halved vs baseline: (hd<64)?16:(1024/hd)
    // smem = 4 * block_n5 * (hd+1) ≈ 16 KB → fits 6 blocks/SM on RTX 3060
#define LAUNCH_MEM_BWD_EXP5(HD) \
    do { \
        const int block_n5 = ((HD) < 64) ? 16 : (1024 / (HD)); \
        const size_t shmem_exp5 = 4ULL * block_n5 * ((HD) + 1) * sizeof(float); \
        const int kv_tiles5 = ((int)T + block_n5 - 1) / block_n5; \
        dim3 grid_bwd5(kv_tiles5, BH); \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp5<HD, true> \
                <<<grid_bwd5, block_cfg, shmem_exp5>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp5<HD, false> \
                <<<grid_bwd5, block_cfg, shmem_exp5>>>(params); \
        } \
    } while (0)

    // Exp6: Global atomics for dK/dV (eliminates dKs/dVs from smem entirely)
    // smem = ONLY Ks + Vs = 2 * block_n6 * (hd+1) ≈ 8.32 KB → fits ~12 blocks/SM
    // Each block owns unique KV rows → no inter-block contention on global atomics
    // Moves serialization from MIO pipeline → L2 subsystem, breaking smem stall ceiling
#define LAUNCH_MEM_BWD_EXP6(HD) \
    do { \
        const int block_n6 = ((HD) < 64) ? 16 : (1024 / (HD)); \
        const size_t shmem_exp6 = 2ULL * block_n6 * ((HD) + 1) * sizeof(float); \
        const int kv_tiles6 = ((int)T + block_n6 - 1) / block_n6; \
        dim3 grid_bwd6(kv_tiles6, BH); \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp6<HD, true> \
                <<<grid_bwd6, block_cfg, shmem_exp6>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp6<HD, false> \
                <<<grid_bwd6, block_cfg, shmem_exp6>>>(params); \
        } \
    } while (0)

    // Exp7: Exp6 + bank-conflict-free K/V smem loading (scalar, not float4)
    // Nsight exp6: Est. Speedup 45.88% from fixing smem store bank conflicts.
    // Same smem, same grid, same BlockN as exp6.
#define LAUNCH_MEM_BWD_EXP7(HD) \
    do { \
        const int block_n7 = ((HD) < 64) ? 16 : (1024 / (HD)); \
        const size_t shmem_exp7 = 2ULL * block_n7 * ((HD) + 1) * sizeof(float); \
        const int kv_tiles7 = ((int)T + block_n7 - 1) / block_n7; \
        dim3 grid_bwd7(kv_tiles7, BH); \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp7<HD, true> \
                <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp7<HD, false> \
                <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
        } \
    } while (0)

    // Exp8: WMMA tensor-core dK/dV + scalar dQ.
    // Only valid for HeadDim % 16 == 0 (WMMA tile requirement).
    // Non-multiples fall back to EXP7.
    // smem = 2*BlockN*(HD+1)*sizeof(float)          [Ks+Vs fp32]
    //      + (2*BM_WMMA*HD + 2*BlockN*BM_WMMA)*sizeof(half) [Q_h+dO_h+ds_T_h+p_T_h fp16]
    // BlockN=16, BM_WMMA=16 (fixed for WMMA).
#define LAUNCH_MEM_BWD_EXP8(HD) \
    do { \
        constexpr int BN8 = 16, BM8 = 16; \
        const size_t shmem8 = 2ULL*BN8*((HD)+1)*sizeof(float) \
                            + (2ULL*BM8*(HD) + 2ULL*BN8*BM8)*sizeof(__half); \
        const int kv8 = ((int)T + BN8 - 1) / BN8; \
        dim3 grid8(kv8, BH); \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp8<HD, true> \
                <<<grid8, block_cfg, shmem8>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp8<HD, false> \
                <<<grid8, block_cfg, shmem8>>>(params); \
        } \
    } while (0)

    // Exp9: TF32 WMMA (fair comparison — fp32 range, matches PyTorch cuBLAS precision).
    // shmem = 4*BlockN*(HD+1)*f32 [Ks+Vs+Q_sm+dO_sm] + 2*BlockN*(BM+1)*f32 [ds_T+p_T]
#define LAUNCH_MEM_BWD_EXP9(HD) \
    do { \
        constexpr int BN9 = 16, BM9 = 16; \
        const size_t shmem9 = 4ULL*BN9*((HD)+1)*sizeof(float) \
                            + 2ULL*BN9*(BM9+1)*sizeof(float); \
        const int kv9 = ((int)T + BN9 - 1) / BN9; \
        dim3 grid9(kv9, BH); \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp9<HD, true> \
                <<<grid9, block_cfg, shmem9>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp9<HD, false> \
                <<<grid9, block_cfg, shmem9>>>(params); \
        } \
    } while (0)

    // Exp10: 32×32 tile TF32 WMMA — optimized for D=64
    // smem = 2*32*(HD+1)*f32 [Ks+Vs] + 2*32*(32+1)*f32 [ds_T+p_T] + 2*32*(HD+1)*f32 [Q_sm+dO_sm]
#define LAUNCH_MEM_BWD_EXP10(HD) \
    do { \
        constexpr int BN10 = 32, BM10 = 32; \
        const size_t shmem10 = (4ULL*BN10*((HD)+1) + 2ULL*BN10*(BM10+1))*sizeof(float); \
        const int kv10 = ((int)T + BN10 - 1) / BN10; \
        dim3 grid10(kv10, BH); \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp10<HD, true> \
                <<<grid10, block_cfg, shmem10>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp10<HD, false> \
                <<<grid10, block_cfg, shmem10>>>(params); \
        } \
    } while (0)

    switch ((int)hd) {
        case   8: LAUNCH_MEM_BWD_EXP7(  8); break;  // not multiple of 16 → exp7
        case  16: LAUNCH_MEM_BWD_EXP9( 16); break;
        case  24: LAUNCH_MEM_BWD_EXP7( 24); break;
        case  32: LAUNCH_MEM_BWD_EXP10(32); break;
        case  40: LAUNCH_MEM_BWD_EXP7( 40); break;
        case  48: LAUNCH_MEM_BWD_EXP9( 48); break;
        case  56: LAUNCH_MEM_BWD_EXP7( 56); break;
        case  64: LAUNCH_MEM_BWD_EXP10(64); break;
        case  80: LAUNCH_MEM_BWD_EXP9( 80); break;
        case  96: LAUNCH_MEM_BWD_EXP9( 96); break;
        case 128: LAUNCH_MEM_BWD_EXP9(128); break;
        case 160: LAUNCH_MEM_BWD_EXP9(160); break;
        case 192: LAUNCH_MEM_BWD_EXP9(192); break;
        case 256: LAUNCH_MEM_BWD_EXP9(256); break;
        default:
            printf("mem_efficient_attn_backward: unsupported head_dim %d\n", (int)hd);
            break;
    }
#undef LAUNCH_MEM_BWD
#undef LAUNCH_MEM_BWD_EXP1
#undef LAUNCH_MEM_BWD_EXP2
#undef LAUNCH_MEM_BWD_EXP3
#undef LAUNCH_MEM_BWD_EXP4
#undef LAUNCH_MEM_BWD_EXP5
#undef LAUNCH_MEM_BWD_EXP6
#undef LAUNCH_MEM_BWD_EXP7
#undef LAUNCH_MEM_BWD_EXP8
#undef LAUNCH_MEM_BWD_EXP9
#undef LAUNCH_MEM_BWD_EXP10
}

} // namespace cuda
} // namespace OwnTensor 