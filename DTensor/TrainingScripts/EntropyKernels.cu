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

} // namespace cuda
} // namespace OwnTensor
