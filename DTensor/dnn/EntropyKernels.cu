#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <cfloat>

namespace OwnTensor {
namespace cuda {

// =============================================================================
// Fused Forward: Computes local_max, local_sum_exp, probs, and target_logit
// in a single kernel launch.  One thread-block per row (B*T rows).
// =============================================================================

__global__ void vocab_parallel_ce_fused_forward_kernel(
    const float* __restrict__ logits,     // [BT, V_local]
    const float* __restrict__ targets,    // [BT]  (float-encoded int targets)
    float* __restrict__ probs,            // [BT, V_local]  output softmax probs
    float* __restrict__ local_max_out,    // [BT]  output per-row max
    float* __restrict__ local_sum_out,    // [BT]  output per-row sum(exp)
    float* __restrict__ target_logit_out, // [BT]  output target logit (0 if not local)
    int64_t V_local,
    int64_t start_v
) {
    int row = blockIdx.x;  // one block per row
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    const float* row_logits = logits + row * V_local;
    float* row_probs = probs + row * V_local;

    // --- Pass 1: find local max ---
    float my_max = -FLT_MAX;
    for (int j = tid; j < V_local; j += blockSize) {
        float v = row_logits[j];
        if (v > my_max) my_max = v;
    }
    // Warp reduction for max
    for (int offset = 16; offset > 0; offset >>= 1)
        my_max = fmaxf(my_max, __shfl_down_sync(0xffffffff, my_max, offset));

    // Block reduction using shared memory (limited to 32 warps = 1024 threads)
    __shared__ float smax[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) smax[warp_id] = my_max;
    __syncthreads();

    // First warp aggregates
    if (warp_id == 0) {
        my_max = (tid < (blockSize + 31) / 32) ? smax[tid] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1)
            my_max = fmaxf(my_max, __shfl_down_sync(0xffffffff, my_max, offset));
    }
    __shared__ float row_max;
    if (tid == 0) row_max = my_max;
    __syncthreads();

    // --- Pass 2: compute exp(logit - max) and sum ---
    float my_sum = 0.0f;
    for (int j = tid; j < V_local; j += blockSize) {
        float e = expf(row_logits[j] - row_max);
        row_probs[j] = e;  // temporary: store exp values
        my_sum += e;
    }
    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset >>= 1)
        my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);

    __shared__ float ssum[32];
    if (lane == 0) ssum[warp_id] = my_sum;
    __syncthreads();

    if (warp_id == 0) {
        my_sum = (tid < (blockSize + 31) / 32) ? ssum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
    }
    __shared__ float row_sum;
    if (tid == 0) {
        row_sum = my_sum;
        local_max_out[row] = row_max;
        local_sum_out[row] = row_sum;
    }
    __syncthreads();

    // --- Pass 3: normalize to get probs (will be corrected after global AllReduce) ---
    // For now store exp values; normalization will happen after global sum_exp is known.
    // Actually, we keep exp_logits in probs for now; the host code will divide after AllReduce.

    // --- Extract target logit ---
    if (tid == 0) {
        int64_t target_idx = static_cast<int64_t>(targets[row]);
        int64_t local_col = target_idx - start_v;
        if (local_col >= 0 && local_col < V_local) {
            target_logit_out[row] = row_logits[local_col];
        } else {
            target_logit_out[row] = 0.0f;
        }
    }
}

void launch_vocab_parallel_ce_fused_forward(
    const float* logits, const float* targets,
    float* probs, float* local_max, float* local_sum_exp, float* target_logit,
    int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream
) {
    int BT = B * T;
    int threads = 256;  // Good for V_local ~25k (each thread handles ~98 elements)
    vocab_parallel_ce_fused_forward_kernel<<<BT, threads, 0, stream>>>(
        logits, targets, probs, local_max, local_sum_exp, target_logit,
        V_local, start_v
    );
}

// =============================================================================
// Normalize probs after global sum_exp is known: probs[i] /= global_sum_exp[row]
// =============================================================================
__global__ void normalize_probs_kernel(
    float* __restrict__ probs,               // [BT, V_local]
    const float* __restrict__ global_sum_exp, // [BT]
    int64_t V_local,
    int64_t total
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int64_t row = idx / V_local;
        probs[idx] /= global_sum_exp[row];
    }
}

void launch_normalize_probs(
    float* probs, const float* global_sum_exp,
    int64_t BT, int64_t V_local, cudaStream_t stream
) {
    int64_t total = BT * V_local;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    normalize_probs_kernel<<<blocks, threads, 0, stream>>>(
        probs, global_sum_exp, V_local, total
    );
}

// =============================================================================
// Rescale probs by per-row factor: probs[i] *= scale[row]
// Used to convert from local_max basis to global_max basis
// =============================================================================
__global__ void rescale_probs_kernel(
    float* __restrict__ probs,         // [BT, V_local]
    const float* __restrict__ scale,   // [BT]
    int64_t V_local,
    int64_t total
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int64_t row = idx / V_local;
        probs[idx] *= scale[row];
    }
}

void launch_rescale_probs(
    float* probs, const float* scale,
    int64_t BT, int64_t V_local, cudaStream_t stream
) {
    int64_t total = BT * V_local;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rescale_probs_kernel<<<blocks, threads, 0, stream>>>(
        probs, scale, V_local, total
    );
}

// Input: logits [B, T, V_local], targets [B, T]
// Output: out [B, T, 1]
// Check if target[b,t] is in [start_v, start_v + V_local). If so, gather logit.
__global__ void extract_target_logits_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ out,
    int64_t B, int64_t T, int64_t V_local,
    int64_t start_v
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t numel = B * T;

    if (idx < numel) {
        // targets should be integer-like, but passed as float
        int64_t target_idx = static_cast<int64_t>(targets[idx]);
        
        // Check if target is in local vocabulary partition
        if (target_idx >= start_v && target_idx < start_v + V_local) {
            int64_t local_col = target_idx - start_v;
            // logits shape is [B*T, V_local] logically
            // logits are contiguous: [row0, row1, ...]
            out[idx] = logits[idx * V_local + local_col];
        } else {
            out[idx] = 0.0f; // Padding or other rank holds the target
        }
    }
}

void launch_extract_target_logits(const float* logits, const float* targets, float* out,
                                  int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream) {
    int64_t numel = B * T;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    extract_target_logits_kernel<<<blocks, threads, 0, stream>>>(
        logits, targets, out, B, T, V_local, start_v
    );
    
    // Check for launch errors (optional, but good for debugging)
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA Error in extract_target_logits: " << cudaGetErrorString(err) << std::endl;
    // }
}

// Input: grad [B, T, V_local], targets [B, T]
// Operation: grad[b, t, target - start_v] -= g_out
// Only if target is in local range.
__global__ void sparse_subtract_kernel(
    float* __restrict__ grad,
    const float* __restrict__ targets,
    float g_out,
    int64_t B, int64_t T, int64_t V_local,
    int64_t start_v
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t numel = B * T;

    if (idx < numel) {
        int64_t target_idx = static_cast<int64_t>(targets[idx]);

        // Check availability in local shard
        if (target_idx >= start_v && target_idx < start_v + V_local) {
            int64_t local_col = target_idx - start_v;
            // Subtract g_out from the specific logit gradient
            grad[idx * V_local + local_col] -= g_out;
        }
    }
}

void launch_sparse_subtract(float* grad, const float* targets, float g_out,
                           int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream) {
    int64_t numel = B * T;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    sparse_subtract_kernel<<<blocks, threads, 0, stream>>>(
        grad, targets, g_out, B, T, V_local, start_v
    );
}

// =============================================================================
// Fused Vocab-Parallel Cross Entropy Backward Kernel
// =============================================================================
// Computes: grad_logits[i, j] = probs[i, j] * (grad_output[0] * scale)
//           grad_logits[i, target_local] -= (grad_output[0] * scale)  (if target in local shard)
// All on-GPU, no CPU sync needed.

__global__ void vocab_parallel_ce_backward_kernel(
    const float* __restrict__ probs,      // [B*T, V_local] softmax probs
    const float* __restrict__ targets,    // [B*T] (float-encoded int targets)
    const float* __restrict__ grad_out,   // [1] scalar on GPU
    float scale,                          // 1.0f / (B * T)
    float* __restrict__ grad_logits,      // [B*T, V_local] output
    int64_t BT,                           // B * T
    int64_t V_local,
    int64_t start_v
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BT * V_local;

    if (idx < total) {
        int64_t row = idx / V_local;
        int64_t col = idx % V_local;

        float g = grad_out[0] * scale;
        float val = probs[idx] * g;

        // Check if this column is the target for this row
        int64_t target_idx = static_cast<int64_t>(targets[row]);
        int64_t local_target = target_idx - start_v;
        if (local_target >= 0 && local_target < V_local && col == local_target) {
            val -= g;
        }

        grad_logits[idx] = val;
    }
}

void launch_vocab_parallel_ce_backward(
    const float* probs, const float* targets, const float* grad_out,
    float scale, float* grad_logits,
    int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream
) {
    int64_t total = B * T * V_local;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    vocab_parallel_ce_backward_kernel<<<blocks, threads, 0, stream>>>(
        probs, targets, grad_out, scale, grad_logits,
        B * T, V_local, start_v
    );
}

} // namespace cuda
} // namespace OwnTensor
