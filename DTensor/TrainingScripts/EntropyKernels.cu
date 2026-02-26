#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

namespace OwnTensor {
namespace cuda {

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
// Fused Forward: sum_exp + target_logit extraction in ONE pass
// =============================================================================
// One thread block per (b,t) position. Block reduction over V_local elements.
// Replaces: exp(logits-max) → reduce_sum → extract_target (3 passes → 1)

__global__ void fused_sum_exp_target_kernel(
    const float* __restrict__ logits,      // [BT * V_local]
    const float* __restrict__ global_max,  // [BT]
    const float* __restrict__ targets,     // [BT] as float
    float* __restrict__ packed_out,        // [BT * 2]: first BT = sum_exp, next BT = target_logit
    int64_t BT, int64_t V_local, int64_t start_v
) {
    extern __shared__ float sdata[];  // blockDim.x floats

    int64_t bt = blockIdx.x;
    if (bt >= BT) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float max_val = global_max[bt];
    int64_t target_idx = (int64_t)targets[bt];
    const float* row = logits + bt * V_local;

    // Phase 1: Each thread accumulates over its stripe of V_local
    float partial_sum = 0.0f;
    float target_logit = 0.0f;

    for (int64_t v = tid; v < V_local; v += blockSize) {
        float logit = row[v];
        partial_sum += expf(logit - max_val);
        if ((v + start_v) == target_idx) {
            target_logit = logit;
        }
    }

    // Phase 2: Block reduction for sum_exp
    sdata[tid] = partial_sum;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) packed_out[bt] = sdata[0];  // sum_exp

    // Phase 3: Reduce target_logit (at most 1 thread has non-zero)
    sdata[tid] = target_logit;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) packed_out[BT + bt] = sdata[0];  // target_logit
}

void launch_fused_sum_exp_target(
    const float* logits, const float* global_max, const float* targets,
    float* packed_out, int64_t B, int64_t T, int64_t V_local,
    int64_t start_v, cudaStream_t stream
) {
    int64_t BT = B * T;
    int threads = 256;
    int shared_mem = threads * sizeof(float);

    fused_sum_exp_target_kernel<<<BT, threads, shared_mem, stream>>>(
        logits, global_max, targets, packed_out, BT, V_local, start_v
    );
}

// =============================================================================
// Fused Backward: softmax + grad_scale + sparse_subtract in ONE kernel
// =============================================================================
// Reads grad_output from device pointer (no cudaMemcpy to host needed)

__global__ void vocab_parallel_ce_backward_kernel(
    const float* __restrict__ logits,       // [BT * V_local]
    const float* __restrict__ targets,      // [BT] as float
    const float* __restrict__ sum_exp,      // [BT]
    const float* __restrict__ max_logits,   // [BT]
    const float* __restrict__ grad_output,  // [1] scalar on GPU
    float* __restrict__ grad_logits,        // [BT * V_local] output
    int64_t BT, int64_t V_local,
    int64_t start_v, float scale
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BT * V_local;

    if (idx < total) {
        int64_t bt = idx / V_local;
        int64_t v  = idx % V_local;

        float g_out = grad_output[0] * scale;

        // Fused softmax: exp(logit - max) / sum_exp
        float softmax_val = expf(logits[idx] - max_logits[bt]) / sum_exp[bt];

        // grad = softmax * g_out
        float grad = softmax_val * g_out;

        // Sparse subtract: if this vocab position is the target
        int64_t target_idx = (int64_t)targets[bt];
        if (target_idx >= start_v && target_idx < start_v + V_local) {
            if (v == (target_idx - start_v)) {
                grad -= g_out;
            }
        }

        grad_logits[idx] = grad;
    }
}

void launch_vocab_parallel_ce_backward(
    const float* logits, const float* targets,
    const float* sum_exp, const float* max_logits,
    const float* grad_output, float* grad_logits,
    int64_t B, int64_t T, int64_t V_local,
    int64_t start_v, float scale, cudaStream_t stream
) {
    int64_t total = B * T * V_local;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    vocab_parallel_ce_backward_kernel<<<blocks, threads, 0, stream>>>(
        logits, targets, sum_exp, max_logits, grad_output, grad_logits,
        B * T, V_local, start_v, scale
    );
}

// =============================================================================
// Fused Vocab-Parallel Embedding Kernels
// =============================================================================
// Forward: replaces 9 separate ops (type cast, compare, mask, subtract, mul, embedding, reshape, mul)
//          with a single kernel: check-in-shard → lookup-or-zero
// Backward: scatter-add upstream gradients into weight.grad at looked-up indices

__global__ void vocab_parallel_embedding_fwd_kernel(
    const int64_t* __restrict__ input,    // [B*T] token indices
    const float*   __restrict__ weight,   // [local_V, D] embedding table
    float*         __restrict__ output,   // [B*T*D] output embeddings
    int64_t BT, int64_t D,
    int64_t start_v, int64_t end_v
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BT * D;

    if (idx < total) {
        int64_t bt = idx / D;
        int64_t d  = idx % D;

        int64_t token = input[bt];

        if (token >= start_v && token < end_v) {
            int64_t local_idx = token - start_v;
            output[idx] = weight[local_idx * D + d];
        } else {
            output[idx] = 0.0f;
        }
    }
}

void launch_vocab_parallel_embedding_fwd(
    const int64_t* input, const float* weight, float* output,
    int64_t B, int64_t T, int64_t D,
    int64_t start_v, int64_t end_v, cudaStream_t stream
) {
    int64_t total = B * T * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    vocab_parallel_embedding_fwd_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, output, B * T, D, start_v, end_v
    );
}

__global__ void vocab_parallel_embedding_bwd_kernel(
    const int64_t* __restrict__ input,        // [B*T] token indices
    const float*   __restrict__ grad_output,  // [B*T*D] upstream grad
    float*         __restrict__ grad_weight,  // [local_V, D] weight grad (accumulate)
    int64_t BT, int64_t D,
    int64_t start_v, int64_t end_v
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BT * D;

    if (idx < total) {
        int64_t bt = idx / D;
        int64_t d  = idx % D;

        int64_t token = input[bt];

        if (token >= start_v && token < end_v) {
            int64_t local_idx = token - start_v;
            atomicAdd(&grad_weight[local_idx * D + d], grad_output[idx]);
        }
    }
}

void launch_vocab_parallel_embedding_bwd(
    const int64_t* input, const float* grad_output, float* grad_weight,
    int64_t B, int64_t T, int64_t D,
    int64_t start_v, int64_t end_v, cudaStream_t stream
) {
    int64_t total = B * T * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    vocab_parallel_embedding_bwd_kernel<<<blocks, threads, 0, stream>>>(
        input, grad_output, grad_weight, B * T, D, start_v, end_v
    );
}

} // namespace cuda
} // namespace OwnTensor
