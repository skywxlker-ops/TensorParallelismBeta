#include "ops/helpers/EmbeddingKernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace OwnTensor {
namespace cuda {

// =============================================================================
// OPTIMIZED EMBEDDING FORWARD KERNEL
// Each thread processes one element of the embedding vector
// This allows for coalesced memory access
// Supports strided weight tensors (e.g., transposed views for weight tying)
// =============================================================================

__global__ void embedding_forward_kernel_optimized(
    const uint16_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t N,      // Number of indices (batch * seq_len)
    int64_t C,      // Embedding dimension
    int64_t V,      // Vocabulary size
    int padding_idx,
    int64_t weight_stride_row,  // stride between rows (tokens) in weight
    int64_t weight_stride_col   // stride between cols (embed dims) in weight
) {
    // 2D thread indexing: x = embedding element, y = index position
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;  // embedding dimension
    int64_t n = blockIdx.y * blockDim.y + threadIdx.y;  // index position
    
    if (n >= N || c >= C) return;
    
    uint16_t tok = indices[n];
    
    if (tok == (uint16_t)padding_idx) {
        output[n * C + c] = 0.0f;
        return;
    }
    
    if (tok >= V) {
        return;
    }
    
    output[n * C + c] = weight[(size_t)tok * weight_stride_row + c * weight_stride_col];
}

// Alternative: Use shared memory for indices when C is small
__global__ void embedding_forward_kernel_small_c(
    const uint16_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx,
    int64_t weight_stride_row,
    int64_t weight_stride_col
) {
    extern __shared__ uint16_t shared_indices[];
    
    int64_t tid = threadIdx.x;
    int64_t bid = blockIdx.x;
    int64_t block_size = blockDim.x;
    
    // Each block handles a portion of indices
    int64_t indices_per_block = block_size;
    int64_t start_n = bid * indices_per_block;
    
    // Load indices to shared memory
    if (start_n + tid < N) {
        shared_indices[tid] = indices[start_n + tid];
    }
    __syncthreads();
    
    // Each thread copies one full embedding row
    int64_t n = start_n + tid;
    if (n >= N) return;
    
    uint16_t tok = shared_indices[tid];
    float* out_row = output + n * C;
    
    if (tok == (uint16_t)padding_idx) {
        for (int64_t j = 0; j < C; ++j) {
            out_row[j] = 0.0f;
        }
        return;
    }
    
    if (tok >= V) return;
    
    // Use strided access for weight
    for (int64_t j = 0; j < C; ++j) {
        out_row[j] = weight[(size_t)tok * weight_stride_row + j * weight_stride_col];
    }
}

// =============================================================================
// OPTIMIZED EMBEDDING BACKWARD KERNEL
// Uses warp-level atomics for better performance
// Supports strided grad_weight tensors
// =============================================================================

__global__ void embedding_backward_kernel_optimized(
    const uint16_t* __restrict__ indices,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weight,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx,
    int64_t grad_weight_stride_row,
    int64_t grad_weight_stride_col
) {
    // 2D thread indexing
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (n >= N || c >= C) return;
    
    uint16_t tok = indices[n];
    if (tok == (uint16_t)padding_idx || tok >= V) return;
    
    float grad = grad_output[n * C + c];
    atomicAdd(&grad_weight[(size_t)tok * grad_weight_stride_row + c * grad_weight_stride_col], grad);
}

void embedding_forward_cuda(
    const uint16_t* indices,
    const float* weight,
    float* output,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx,
    int64_t weight_stride_row,
    int64_t weight_stride_col
) {
    // Use 2D grid for better parallelism
    dim3 block(32, 8);  // 256 threads total
    dim3 grid(
        (C + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );
    
    embedding_forward_kernel_optimized<<<grid, block>>>(
        indices, weight, output, N, C, V, padding_idx,
        weight_stride_row, weight_stride_col
    );
}

void embedding_backward_cuda(
    const uint16_t* indices,
    const float* grad_output,
    float* grad_weight,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx,
    int64_t grad_weight_stride_row,
    int64_t grad_weight_stride_col
) {
    // Use 2D grid for better parallelism
    dim3 block(32, 8);  // 256 threads total
    dim3 grid(
        (C + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );
    
    embedding_backward_kernel_optimized<<<grid, block>>>(
        indices, grad_output, grad_weight, N, C, V, padding_idx,
        grad_weight_stride_row, grad_weight_stride_col
    );
}

} // namespace cuda
} // namespace OwnTensor