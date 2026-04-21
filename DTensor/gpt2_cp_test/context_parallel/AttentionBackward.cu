#include "ops/cuda/attention/AttentionCommon.cuh"
#include "ops/helpers/AttentionKernels.h"
#include "ops/helpers/KernelDispatch.h"
#include "autograd/backward/AttentionBackward.h"
#include "device/DeviceCore.h"

// ============================================================================
// Context-Parallel aware copy of Tensor-Implementations AttentionBackward.
//
// Adds separate T_q / T_k plus q_offset / k_offset for CP ring-attention
// sub-chunks.
//
// Key behavior change from TI:
//   - exp11's final dQ write is atomicAdd (not assignment). Callers must zero
//     dQ once before the first ring iteration; subsequent iterations accumulate.
//   - dK/dV are treated as per-iteration outputs (zeroed on entry as in TI).
//
// Wrapped in namespace OwnTensor::cp to avoid ODR conflicts with TI.
// ============================================================================

namespace OwnTensor {
namespace cp {

// ============================================================================
// CP backward params
// ============================================================================

struct CPBwdParams {
    const float* Q;
    const float* K;
    const float* V;
    const float* dO;
    const float* LSE;
    const float* D;
    float* dQ;
    float* dK;
    float* dV;
    int T_q;
    int T_k;
    int q_offset;
    int k_offset;
    float scale;
    bool is_causal;
};

// ============================================================================
// Common helpers
// ============================================================================

static constexpr int BWD_BLOCK_M     = 8;
static constexpr int BWD_WARP_SZ     = 32;
static constexpr int BWD_NUM_THREADS = BWD_BLOCK_M * BWD_WARP_SZ;   // 256
static constexpr int BWD_BLOCK_M_D   = 8;

__inline__ __device__ float bwd_warp_sum(float val) {
    #pragma unroll
    for (int offset = BWD_WARP_SZ / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// Precompute D[bh, qi] = sum_d dO[bh, qi, d] * O[bh, qi, d]
// Only uses Q-side, so indexed by T_q.
// ============================================================================

template<int HeadDim>
__global__ void mem_efficient_bwd_precompute_D(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    int T_q)
{
    constexpr int LocalN = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;

    const int bh      = blockIdx.y;
    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const int base_row = blockIdx.x * (BWD_BLOCK_M_D * 2) + warp_id * 2;
    const int row0     = base_row;
    const int row1     = base_row + 1;
    const bool v0      = (row0 < T_q);
    const bool v1      = (row1 < T_q);

    const long long bh_base = (long long)bh * T_q * HeadDim;
    const long long off0    = bh_base + (long long)row0 * HeadDim;
    const long long off1    = bh_base + (long long)row1 * HeadDim;

    float sum0 = 0.f, sum1 = 0.f;

    #pragma unroll
    for (int i = 0; i < LocalN; ++i) {
        const int k = lane_id + i * BWD_WARP_SZ;
        if (k < HeadDim) {
            if (v0) sum0 += __ldg(&dO[off0 + k]) * __ldg(&O[off0 + k]);
            if (v1) sum1 += __ldg(&dO[off1 + k]) * __ldg(&O[off1 + k]);
        }
    }

    sum0 = bwd_warp_sum(sum0);
    sum1 = bwd_warp_sum(sum1);

    if (lane_id == 0) {
        if (v0) D[(long long)bh * T_q + row0] = sum0;
        if (v1) D[(long long)bh * T_q + row1] = sum1;
    }
}

// ============================================================================
// exp7: scalar KV-outer backward (CP)
// - KV tile over T_k, Q loop over T_q
// - dQ, dK, dV all via atomicAdd (dK/dV zeroed by this kernel for its tile)
// - Causal uses global positions via q_offset / k_offset
// ============================================================================

template<int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp7(CPBwdParams params)
{
    constexpr int BlockN  = (HeadDim < 64) ? 16 : (1024 / HeadDim);
    constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD  = HeadDim + 1;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = Ks + BlockN * HD_PAD;

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, params.T_k - tile_start);
    if (tile_start >= params.T_k) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;

    const long long q_bh_off = (long long)bh * params.T_q * HeadDim;
    const long long k_bh_off = (long long)bh * params.T_k * HeadDim;
    const long long q_bh_T   = (long long)bh * params.T_q;

    const float* Q_bh   = params.Q   + q_bh_off;
    const float* K_bh   = params.K   + k_bh_off;
    const float* V_bh   = params.V   + k_bh_off;
    const float* dO_bh  = params.dO  + q_bh_off;
    const float* LSE_bh = params.LSE + q_bh_T;
    const float* D_bh   = params.D   + q_bh_T;
    float*       dQ_bh  = params.dQ  + q_bh_off;
    float*       dK_bh  = params.dK  + k_bh_off;
    float*       dV_bh  = params.dV  + k_bh_off;

    // Zero this block's dK/dV tile rows (T_k-indexed)
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim;
        const int k = idx % HeadDim;
        dK_bh[(tile_start + r) * HeadDim + k] = 0.f;
        dV_bh[(tile_start + r) * HeadDim + k] = 0.f;
    }

    // K/V smem load
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        const int r     = idx / HeadDim;
        const int k     = idx % HeadDim;
        const int g_row = tile_start + r;
        Ks[r * HD_PAD + k] = (g_row < params.T_k) ? K_bh[g_row * HeadDim + k] : 0.f;
        Vs[r * HD_PAD + k] = (g_row < params.T_k) ? V_bh[g_row * HeadDim + k] : 0.f;
    }

    __syncthreads();

    // Causal Q-start in LOCAL coords:
    //   first valid qi (global) = earliest K_global in tile = k_offset + tile_start
    //   first valid qi (local)  = max(0, k_offset + tile_start - q_offset)
    const int q_start = Causal
        ? max(0, params.k_offset + tile_start - params.q_offset)
        : 0;

    for (int q_base = q_start; q_base < params.T_q; q_base += BWD_BLOCK_M) {
        const int qi     = q_base + warp_id;
        const bool valid = (qi < params.T_q);

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

            // Causal mask in global coords
            const int q_global_pos = params.q_offset + qi;
            const int k_global_pos = params.k_offset + tile_start + j;

            float p = 0.0f;
            if (j_valid && valid && !(Causal && k_global_pos > q_global_pos))
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
    }
}

// ============================================================================
// exp11: Q-tile-centric backward, TF32 WMMA (CP)
//
// Differences from TI exp11:
//   - Separate T_q / T_k + q_offset / k_offset (stored in CPBwdParams)
//   - Final dQ write uses atomicAdd so multiple ring iterations can accumulate
//     into the same dQ buffer. Caller must zero dQ once before the first call.
//   - dK/dV are written via atomicAdd as in TI; caller zeros the dK/dV buffers
//     for each ring iteration (done in launch macro).
// ============================================================================

template <int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp11(CPBwdParams params)
{
    using namespace nvcuda;

    constexpr int BlockN    = 16;
    constexpr int BM_WMMA   = 16;
    constexpr int HD_CHUNKS = HeadDim / 16;
    constexpr int HD_PAD    = HeadDim;
    constexpr int BKN_PAD   = BlockN;
    constexpr int BM_PAD    = BM_WMMA;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    extern __shared__ float smem_f[];
    float* Q_sm    = smem_f;
    float* dO_sm   = Q_sm    + BM_WMMA * HD_PAD;
    float* Ks      = dO_sm   + BM_WMMA * HD_PAD;
    float* Vs      = Ks      + BlockN   * HD_PAD;
    float* ds_qd   = Vs      + BlockN   * HD_PAD;
    float* DPV_sm  = ds_qd   + BM_WMMA  * BKN_PAD;
    float* ds_kd   = DPV_sm  + BM_WMMA  * BKN_PAD;
    float* p_kd    = ds_kd   + BlockN   * BM_PAD;
    float* tile_st = p_kd    + BlockN   * BM_PAD;
    float* LSE_sm  = tile_st + BlockN   * HeadDim;
    float* D_sm    = LSE_sm  + BM_WMMA;

    const int bh           = blockIdx.y;
    const int q_tile       = blockIdx.x;
    const int q_tile_start = q_tile * BM_WMMA;
    const int tile_size    = min(BM_WMMA, params.T_q - q_tile_start);
    if (q_tile_start >= params.T_q) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int chunk   = warp_id % HD_CHUNKS;

    const long long q_bh_off = (long long)bh * params.T_q * HeadDim;
    const long long k_bh_off = (long long)bh * params.T_k * HeadDim;
    const long long q_bh_T   = (long long)bh * params.T_q;

    const float* Q_bh   = params.Q   + q_bh_off;
    const float* K_bh   = params.K   + k_bh_off;
    const float* V_bh   = params.V   + k_bh_off;
    const float* dO_bh  = params.dO  + q_bh_off;
    const float* LSE_bh = params.LSE + q_bh_T;
    const float* D_bh   = params.D   + q_bh_T;
    float*       dQ_bh  = params.dQ  + q_bh_off;
    float*       dK_bh  = params.dK  + k_bh_off;
    float*       dV_bh  = params.dV  + k_bh_off;

    // Load Q, dO, LSE, D for this Q-tile into smem
    for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
        const int r  = idx / HeadDim, k = idx % HeadDim;
        const int qi = q_tile_start + r;
        const bool vq = (qi < params.T_q);
        Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * HeadDim + k] : 0.f;
        dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * HeadDim + k] : 0.f;
    }
    if (threadIdx.x < BM_WMMA) {
        const int qi  = q_tile_start + threadIdx.x;
        const bool vq = (qi < params.T_q);
        LSE_sm[threadIdx.x] = vq ? LSE_bh[qi] : 0.f;
        D_sm  [threadIdx.x] = vq ? D_bh  [qi] : 0.f;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> dq_frag;
    wmma::fill_fragment(dq_frag, 0.0f);

    // Causal KV loop end in global coords, then clamped to [0, T_k]
    int kv_loop_end;
    if (Causal) {
        int ke = params.q_offset + q_tile_start + BM_WMMA - params.k_offset;
        if (ke < 0) ke = 0;
        if (ke > params.T_k) ke = params.T_k;
        kv_loop_end = ke;
    } else {
        kv_loop_end = params.T_k;
    }

    for (int kv_base = 0; kv_base < kv_loop_end; kv_base += BlockN) {
        __syncthreads();

        const int kv_tile_size = min(BlockN, params.T_k - kv_base);

        // Load K/V tile for this KV iteration
        for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            const int g = kv_base + r;
            Ks[r * HD_PAD + k] = (g < params.T_k) ? K_bh[g * HeadDim + k] : 0.f;
            Vs[r * HD_PAD + k] = (g < params.T_k) ? V_bh[g * HeadDim + k] : 0.f;
        }
        __syncthreads();

        // Phase A: TF32 WMMA GEMMs
        if (warp_id < 2) {
            const float* src_sm = (warp_id == 0) ? Q_sm   : dO_sm;
            const float* kv_sm  = (warp_id == 0) ? Ks     : Vs;
            float*       dst_sm = (warp_id == 0) ? ds_qd  : DPV_sm;

            wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
            wmma::fill_fragment(acc_frag, 0.0f);

            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::col_major> b_frag;

            #pragma unroll
            for (int ks = 0; ks < 2 * HD_CHUNKS; ++ks) {
                const int k_off = ks * 8;
                wmma::load_matrix_sync(a_frag, src_sm + k_off, HD_PAD);
                wmma::load_matrix_sync(b_frag, kv_sm  + k_off, HD_PAD);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            wmma::store_matrix_sync(dst_sm, acc_frag, BKN_PAD, wmma::mem_row_major);
        }
        __syncthreads();

        // Post-process
        {
            const int qi_local  = threadIdx.x / BlockN;
            const int j_local   = threadIdx.x % BlockN;
            const int qi_row    = q_tile_start + qi_local;
            const int j_row     = kv_base + j_local;
            const int qi_global = params.q_offset + qi_row;
            const int j_global  = params.k_offset + j_row;

            const float raw_s = ds_qd [qi_local * BKN_PAD + j_local];
            const float dpv   = DPV_sm[qi_local * BKN_PAD + j_local];
            const float L     = LSE_sm[qi_local];
            const float D_    = D_sm  [qi_local];

            const bool qi_valid  = (qi_row < params.T_q);
            const bool j_valid   = (j_local < kv_tile_size);
            const bool causal_ok = !Causal || (j_global <= qi_global);

            float p = 0.f;
            if (qi_valid && j_valid && causal_ok)
                p = exp2f(BWD_LOG2E * (raw_s * params.scale - L));

            const float ds = p * (dpv - D_) * params.scale;

            ds_qd[qi_local * BKN_PAD + j_local]  = ds;
            ds_kd[j_local  * BM_PAD  + qi_local] = ds;
            p_kd [j_local  * BM_PAD  + qi_local] = p;
        }
        __syncthreads();

        // Phase B.1: dQ WMMA (warps 0..HD_CHUNKS-1) + dK WMMA (warps HD_CHUNKS..)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            if (warp_id < HD_CHUNKS) {
                const float* b_ptr = Ks + chunk * 16;
                wmma::load_matrix_sync(a_frag, ds_qd,              BKN_PAD);
                wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);
                wmma::mma_sync(dq_frag, a_frag, b_frag, dq_frag);
                wmma::load_matrix_sync(a_frag, ds_qd + 8,          BKN_PAD);
                wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
                wmma::mma_sync(dq_frag, a_frag, b_frag, dq_frag);
            } else {
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> dk_frag;
                wmma::fill_fragment(dk_frag, 0.0f);
                const float* b_ptr = Q_sm + chunk * 16;
                wmma::load_matrix_sync(a_frag, ds_kd,              BM_PAD);
                wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);
                wmma::mma_sync(dk_frag, a_frag, b_frag, dk_frag);
                wmma::load_matrix_sync(a_frag, ds_kd + 8,          BM_PAD);
                wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
                wmma::mma_sync(dk_frag, a_frag, b_frag, dk_frag);
                wmma::store_matrix_sync(tile_st + chunk * 16, dk_frag,
                                        HeadDim, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // atomicAdd dK tile -> global
        for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            atomicAdd(&dK_bh[(kv_base + r) * HeadDim + k], tile_st[r * HeadDim + k]);
        }
        __syncthreads();

        // Phase B.2: dV WMMA (warps HD_CHUNKS..)
        if (warp_id >= HD_CHUNKS) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> dv_frag;
            wmma::fill_fragment(dv_frag, 0.0f);

            const float* b_ptr = dO_sm + chunk * 16;
            wmma::load_matrix_sync(a_frag, p_kd,              BM_PAD);
            wmma::load_matrix_sync(b_frag, b_ptr,             HD_PAD);
            wmma::mma_sync(dv_frag, a_frag, b_frag, dv_frag);
            wmma::load_matrix_sync(a_frag, p_kd + 8,          BM_PAD);
            wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
            wmma::mma_sync(dv_frag, a_frag, b_frag, dv_frag);
            wmma::store_matrix_sync(tile_st + chunk * 16, dv_frag,
                                    HeadDim, wmma::mem_row_major);
        }
        __syncthreads();

        // atomicAdd dV tile -> global
        for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            atomicAdd(&dV_bh[(kv_base + r) * HeadDim + k], tile_st[r * HeadDim + k]);
        }
    }

    // Final dQ: atomicAdd so CP ring iterations accumulate into a shared buffer.
    __syncthreads();

    if (warp_id < HD_CHUNKS) {
        wmma::store_matrix_sync(tile_st + chunk * 16, dq_frag,
                                HeadDim, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        atomicAdd(&dQ_bh[(q_tile_start + r) * HeadDim + k], tile_st[r * HeadDim + k]);
    }
}

// ============================================================================
// Public API (CP) — namespace OwnTensor::cp::cuda
// ============================================================================

namespace cuda {

void mem_efficient_attn_backward(
    const float* query, const float* key, const float* value,
    const float* output, const float* grad_output, const float* lse,
    float* grad_query, float* grad_key, float* grad_value,
    float* D_buf,
    int64_t B, int64_t nh,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,
    int64_t hd,
    bool is_causal)
{
    float scale = 1.0f / sqrtf(static_cast<float>(hd));
    const int BH = (int)(B * nh);
    dim3 block_cfg(BWD_NUM_THREADS);

    dim3 grid_D(((int)T_q + (BWD_BLOCK_M_D * 2) - 1) / (BWD_BLOCK_M_D * 2), BH);

    CPBwdParams params;
    params.Q         = query;
    params.K         = key;
    params.V         = value;
    params.dO        = grad_output;
    params.LSE       = lse;
    params.D         = D_buf;
    params.dQ        = grad_query;
    params.dK        = grad_key;
    params.dV        = grad_value;
    params.T_q       = (int)T_q;
    params.T_k       = (int)T_k;
    params.q_offset  = q_offset;
    params.k_offset  = k_offset;
    params.scale     = scale;
    params.is_causal = is_causal;

    // exp7: scalar KV-outer fallback (any HeadDim, uses atomicAdd for dQ/dK/dV)
    #define LAUNCH_CP_BWD_EXP7(HD) \
    do { \
        const int block_n7 = ((HD) < 64) ? 16 : (1024 / (HD)); \
        const size_t shmem_exp7 = 2ULL * block_n7 * ((HD) + 1) * sizeof(float); \
        const int kv_tiles7 = ((int)T_k + block_n7 - 1) / block_n7; \
        dim3 grid_bwd7(kv_tiles7, BH); \
        ::OwnTensor::cp::mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T_q); \
        if (is_causal) { \
            ::OwnTensor::cp::mem_efficient_bwd_unified_kernel_exp7<HD, true> \
                <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
        } else { \
            ::OwnTensor::cp::mem_efficient_bwd_unified_kernel_exp7<HD, false> \
                <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
        } \
    } while (0)

    // exp11: Q-tile-centric, TF32 WMMA, dQ/dK/dV via atomicAdd (HD%16==0)
    #define LAUNCH_CP_BWD_EXP11(HD) \
    do { \
        constexpr int BN11 = 16, BM11 = 16; \
        const size_t shmem11 = \
            (2ULL * BM11 * ((HD) + 1) \
           + 2ULL * BN11 * ((HD) + 1) \
           + 2ULL * BM11 * ((BN11) + 1) \
           + 2ULL * BN11 * ((BM11) + 1) \
           + 1ULL * BN11 * (HD) \
           + 2ULL * BM11 \
            ) * sizeof(float); \
        const int q11 = ((int)T_q + BM11 - 1) / BM11; \
        dim3 grid_q11(q11, BH); \
        cudaFuncSetAttribute( \
            ::OwnTensor::cp::mem_efficient_bwd_unified_kernel_exp11<HD, false>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem11); \
        cudaFuncSetAttribute( \
            ::OwnTensor::cp::mem_efficient_bwd_unified_kernel_exp11<HD, true>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem11); \
        cudaStream_t stream = ::OwnTensor::cuda::getCurrentStream(); \
        cudaMemsetAsync(params.dQ, 0, (size_t)BH * (int)T_q * (HD) * sizeof(float), stream); \
        cudaMemsetAsync(params.dK, 0, (size_t)BH * (int)T_k * (HD) * sizeof(float), stream); \
        cudaMemsetAsync(params.dV, 0, (size_t)BH * (int)T_k * (HD) * sizeof(float), stream); \
        ::OwnTensor::cp::mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T_q); \
        if (is_causal) { \
            ::OwnTensor::cp::mem_efficient_bwd_unified_kernel_exp11<HD, true> \
                <<<grid_q11, block_cfg, shmem11>>>(params); \
        } else { \
            ::OwnTensor::cp::mem_efficient_bwd_unified_kernel_exp11<HD, false> \
                <<<grid_q11, block_cfg, shmem11>>>(params); \
        } \
    } while (0)

    switch ((int)hd) {
        case   8: LAUNCH_CP_BWD_EXP7(  8); break;
        case  16: LAUNCH_CP_BWD_EXP11( 16); break;
        case  24: LAUNCH_CP_BWD_EXP7( 24); break;
        case  32: LAUNCH_CP_BWD_EXP11( 32); break;
        case  40: LAUNCH_CP_BWD_EXP7( 40); break;
        case  48: LAUNCH_CP_BWD_EXP11( 48); break;
        case  56: LAUNCH_CP_BWD_EXP7( 56); break;
        case  64: LAUNCH_CP_BWD_EXP11( 64); break;
        case  80: LAUNCH_CP_BWD_EXP11( 80); break;
        case  96: LAUNCH_CP_BWD_EXP11( 96); break;
        case 128: LAUNCH_CP_BWD_EXP11(128); break;
        case 160: LAUNCH_CP_BWD_EXP11(160); break;
        case 192: LAUNCH_CP_BWD_EXP11(192); break;
        case 256: LAUNCH_CP_BWD_EXP11(256); break;
        default:
            printf("cp::mem_efficient_attn_backward: unsupported head_dim %d\n", (int)hd);
            break;
    }
    #undef LAUNCH_CP_BWD_EXP7
    #undef LAUNCH_CP_BWD_EXP11
}

} // namespace cuda
} // namespace cp
} // namespace OwnTensor
