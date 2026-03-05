#include "ops/helpers/LossKernels.h"
#include "dtype/Types.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <type_traits>
#include "device/CachingCudaAllocator.h"

namespace OwnTensor {
namespace cuda {

// ============================================================================
// Forward Pass CUDA Kernels
// ============================================================================

/**
 * @brief Compute sparse cross entropy loss for each sample (forward pass)
 * 
 * Uses numerically stable log-softmax:
 * loss[i] = log(sum(exp(logits[i] - max(logits[i])))) + max(logits[i]) - logits[i, target[i]]
 */
template<typename T, typename T_idx>
__global__ void sparse_ce_forward_kernel_typed(
    const T* logits,
    const T_idx* targets,
    T* losses,
    int64_t batch_size,
    int64_t vocab_size
) {
    int64_t i = blockIdx.x;
    if (i >= batch_size) return;

    const T* row_logits = logits + i * vocab_size;

    //* Online softmax
    float local_max = -1e38f;
    float local_sum = 0.0f;

    #pragma unroll 4
    for(int64_t j = threadIdx.x; j < vocab_size; j += blockDim.x){
        float val = static_cast<float>(row_logits[j]);
        if(val > local_max){
            //* Update sum based on new max: current_sum * exp(old_max - new_max) + 1.0f
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else{
            //* Standard sum accumulation
            local_sum += expf(val - local_max);
        }
    }

    //* Block reduction in shared memory
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int bdim = blockDim.x;

    //* Use two halves of shared memory: one for max and one for sum
    float *smax = sdata;
    float *ssum = sdata + bdim;

    smax[tid] = local_max;
    ssum[tid] = local_sum;
    __syncthreads(); //* Barrier synchronization

    //* Standard reduction loop
    for(unsigned int s = bdim/2; s > 0; s >>= 1){
        if(tid < s){
            float other_max = smax[tid + s];
            float other_sum = ssum[tid + s];

            //* Merge logic (Online softmax)
            if(other_max > smax[tid]){
                ssum[tid] = ssum[tid] * expf(smax[tid] - other_max) + other_sum;
                smax[tid] = other_max;
            } else{
                ssum[tid] += other_sum * expf(other_max - smax[tid]);
            }
        }
        __syncthreads();
    }

    //* Now index 0 in the shared mem contains the block wide max and sum
    float final_max = smax[0];
    float final_sum = ssum[0];

    //* Final loss computation
    if(tid == 0){
        int64_t target_idx = static_cast<int64_t>(targets[i]);
        float target_logit = static_cast<float>(row_logits[target_idx]);

        //* Loss = log(sum(exp(logits - max))) + max - target_logit
        float loss = logf(final_sum) + final_max - target_logit;
        losses[i] = static_cast<T>(loss);
    }
}

/**
 * @brief Parallel reduction to sum all losses
 * Uses shared memory for efficient reduction
 */
template<typename T>
__global__ void sum_reduction_kernel(
    const T* input,
    T* output,
    int64_t n
) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? static_cast<float>(input[i]) : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    #pragma unroll 4
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = static_cast<T>(sdata[0]);
    }
}

template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
) {
    if (batch_size == 0) {
        // Set loss to 0
        cudaMemsetAsync(loss_output, 0, sizeof(T), stream);
        return;
    }
    
    // Allocate temporary buffer for per-sample losses
    T* d_losses = static_cast<T*>(CachingCUDAAllocator::instance().allocate(batch_size * sizeof(T), stream));
    
    // Kernel 1: Compute loss for each sample
    //* updated the launch config
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem = 2 * threads * sizeof(float);
    sparse_ce_forward_kernel_typed<T, T_idx><<<blocks, threads, shared_mem, stream>>>(
        logits, targets, d_losses, batch_size, vocab_size
    );
    
    // Kernel 2: Reduce to sum all losses
    // For simplicity, use a two-pass reduction
    if (batch_size <= 1024) {
        // Single block reduction
        int reduce_threads = 256;
        int reduce_blocks = 1;
        sum_reduction_kernel<T><<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float), stream>>>(
            d_losses, loss_output, batch_size
        );
    } else {
        // Two-pass reduction for large batches
        int reduce_threads = 256;
        int reduce_blocks = (batch_size + reduce_threads - 1) / reduce_threads;
        
        T* d_partial = static_cast<T*>(CachingCUDAAllocator::instance().allocate(reduce_blocks * sizeof(T), stream));
        
        // First reduction
        sum_reduction_kernel<T><<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float), stream>>>(
            d_losses, d_partial, batch_size
        );
        
        // Second reduction
        sum_reduction_kernel<T><<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
            d_partial, loss_output, reduce_blocks
        );
        
        CachingCUDAAllocator::instance().deallocate(d_partial);
    }
    
    CachingCUDAAllocator::instance().deallocate(d_losses);
}

template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
) {
    sparse_cross_entropy_forward_cuda_impl<T, T_idx>(
        logits, targets, loss_output, batch_size, vocab_size, stream
    );
}

// ============================================================================
// Backward Pass CUDA Kernels (existing)
// ============================================================================

/**
 * @brief CUDA kernel for sparse cross entropy backward pass
 * 
 * Gradient formula (derived from calculus):
 *   For class j:
 *     grad[j] = softmax(logits)[j] * scale           if j != target
 *     grad[j] = (softmax(logits)[j] - 1.0) * scale   if j == target
 * 
 * This is equivalent to: grad = (softmax - target_indicator) * scale
 * where target_indicator equals 1 only at the target class index.
 * 
 * @param logits     Input logits [batch_size, vocab_size]
 * @param targets    Sparse target indices [batch_size] (NOT one-hot!)
 * @param grad       Output gradient [batch_size, vocab_size]
 * @param batch_size Number of samples
 * @param vocab_size Number of classes
 * @param scale      Gradient scaling factor (typically 1/batch_size)
 */
// template<typename T, typename T_idx>
// __global__ void sparse_ce_backward_kernel_typed(
//     const T* logits,
//     const T_idx* targets,
//     T* grad,
//     int64_t batch_size,
//     int64_t vocab_size,
//     T scale
// ) {
//     int64_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (sample_idx >= batch_size) return;

//     // Get pointer to this sample's logits and gradients
//     const T* sample_logits = logits + sample_idx * vocab_size;
//     T* sample_grad = grad + sample_idx * vocab_size;
    
//     // Get the target class for this sample (sparse index, not one-hot)
//     int64_t target_class = static_cast<int64_t>(targets[sample_idx]);
    
//     // ========================================================================
//     // Step 1: Find max logit for numerical stability
//     // ========================================================================
//     float max_logit = -1e38f;
//     for (int64_t c = 0; c < vocab_size; ++c) {
//         float logit_val = static_cast<float>(sample_logits[c]);
//         if (logit_val > max_logit) max_logit = logit_val;
//     }
    
//     // ========================================================================
//     // Step 2: Compute sum of exp(logits - max) for softmax denominator
//     // ========================================================================
//     float sum_exp = 0.0f;
//     for (int64_t c = 0; c < vocab_size; ++c) {
//         sum_exp += expf(static_cast<float>(sample_logits[c]) - max_logit);
//     }
    
//     // ========================================================================
//     // Step 3: Compute gradient for each class
//     // 
//     // The gradient derivation:
//     //   Loss = -log(p_target) where p = softmax(logits)
//     //   
//     //   For non-target classes (c != target):
//     //     dL/d(logit_c) = p_c
//     //   
//     //   For target class (c == target):
//     //     dL/d(logit_c) = p_c - 1
//     //
//     // This can be written as: grad_c = p_c - (c == target ? 1 : 0)
//     // ========================================================================
//     float f_scale = static_cast<float>(scale);
    
//     for (int64_t c = 0; c < vocab_size; ++c) {
//         // Compute softmax probability for this class
//         float prob = expf(static_cast<float>(sample_logits[c]) - max_logit) / sum_exp;
        
//         // Compute gradient:
//         // - For non-target classes: grad = prob * scale
//         // - For target class: grad = (prob - 1) * scale
//         float grad_val;
//         if (c == target_class) {
//             grad_val = (prob - 1.0f) * f_scale;  // Target class: subtract 1
//         } else {
//             grad_val = prob * f_scale;           // Other classes: just softmax
//         }
        
//         sample_grad[c] = static_cast<T>(grad_val);
//     }
// }

template<typename T, typename T_idx>
__global__ void sparse_ce_backward_kernel_typed(
    const T* logits,
    const T_idx* targets,
    T* grad,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale
) {
    int64_t i = blockIdx.x; // One block per sample
    if (i >= batch_size) return;

    // Shared memory for block reduction: blockDim.x floats
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int bdim = blockDim.x;

    const T* row_logits = logits + i * vocab_size;
    T* row_grad = grad + i * vocab_size;
    int64_t target_idx = static_cast<int64_t>(targets[i]);

    // 1. Online Softmax Pass
    // We compute both Max and Sum-of-Exponents in a single pass over memory
    float local_max = -1e38f;
    float local_sum = 0.0f;

    #pragma unroll 4
    for (int64_t j = tid; j < vocab_size; j += bdim) {
        float val = static_cast<float>(row_logits[j]);
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    // Phase 1 Reduction: Reduce Max and Sum within the block
    sdata[tid] = local_max;
    float* ssum = sdata + bdim; // Use second half of shared memory
    ssum[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other_max = sdata[tid + s];
            float other_sum = ssum[tid + s];
            
            if (other_max > sdata[tid]) {
                ssum[tid] = ssum[tid] * expf(sdata[tid] - other_max) + other_sum;
                sdata[tid] = other_max;
            } else {
                ssum[tid] += other_sum * expf(other_max - sdata[tid]);
            }
        }
        __syncthreads();
    }

    float max_val = sdata[0];
    float sum_exp = ssum[0];
    if (sum_exp < 1e-20f) sum_exp = 1e-20f;

    // 2. Compute Gradients
    float f_scale = static_cast<float>(*grad_output) * host_scale;
    float inv_sum = 1.0f / sum_exp;

    #pragma unroll 4
    for (int64_t j = tid; j < vocab_size; j += bdim) {
        float val = static_cast<float>(row_logits[j]);
        float prob = expf(val - max_val) * inv_sum;
        
        float grad_val = (j == target_idx) ? (prob - 1.0f) * f_scale : prob * f_scale;
        row_grad[j] = static_cast<T>(grad_val);
    }
}


// Specialization/Guard for non-numeric types if needed, but static_cast<float> should work for most primitives.
// Complex types will still fail to static_cast to float.

// We will restrict the instantiations to real numeric types.

template<typename T, typename T_idx>
void sparse_cross_entropy_backward_cuda_impl(
    const T* logits,
    const T_idx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    cudaStream_t stream
) {
    if (batch_size == 0) return;
    int threads = 256;
    int blocks = batch_size; // One block per sample

    sparse_ce_backward_kernel_typed<T, T_idx><<<blocks, threads, 2 * threads * sizeof(float), stream>>>(
        logits, targets, grad_logits, batch_size, vocab_size, grad_output, host_scale
    );
}

template<typename T, typename T_idx>
void sparse_cross_entropy_backward_cuda(
    const T* logits,
    const T_idx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    cudaStream_t stream
) {
    sparse_cross_entropy_backward_cuda_impl<T, T_idx>(logits, targets, grad_logits, batch_size, vocab_size, grad_output, host_scale, stream);
}

// Explicit instantiations for supported types
#define INSTANTIATE_GIVEN_T(T) \
    template void sparse_cross_entropy_backward_cuda<T, uint8_t>(const T*, const uint8_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, uint16_t>(const T*, const uint16_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, uint32_t>(const T*, const uint32_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, uint64_t>(const T*, const uint64_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int8_t>(const T*, const int8_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int16_t>(const T*, const int16_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int32_t>(const T*, const int32_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int64_t>(const T*, const int64_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t);

INSTANTIATE_GIVEN_T(float)
INSTANTIATE_GIVEN_T(double)
INSTANTIATE_GIVEN_T(float16_t)
INSTANTIATE_GIVEN_T(bfloat16_t)

// Explicit instantiations for forward pass
#define INSTANTIATE_FORWARD_GIVEN_T(T) \
    template void sparse_cross_entropy_forward_cuda<T, uint8_t>(const T*, const uint8_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, uint16_t>(const T*, const uint16_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, uint32_t>(const T*, const uint32_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, uint64_t>(const T*, const uint64_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int8_t>(const T*, const int8_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int16_t>(const T*, const int16_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int32_t>(const T*, const int32_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int64_t>(const T*, const int64_t*, T*, int64_t, int64_t, cudaStream_t);

INSTANTIATE_FORWARD_GIVEN_T(float)
INSTANTIATE_FORWARD_GIVEN_T(double)
INSTANTIATE_FORWARD_GIVEN_T(float16_t)
INSTANTIATE_FORWARD_GIVEN_T(bfloat16_t)


// ============================================================================
// Categorical Cross Entropy Extensions
// ============================================================================

__global__ void cce_forward_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ losses, // [N]
    int64_t batch_size,
    int64_t num_classes
) {
    int64_t i = blockIdx.x; // One block per sample
    if (i >= batch_size) return;

    const float epsilon = 1e-7f;
    const float one_minus_epsilon = 1.0f - 1e-7f;

    const float* row_pred = predictions + i * num_classes;
    const float* row_targ = targets + i * num_classes;

    float sum = 0.0f;
    #pragma unroll 4
    for (int64_t j = threadIdx.x; j < num_classes; j += blockDim.x) {
        float p = row_pred[j];
        float t = row_targ[j];
        
        // Clip
        if (p < epsilon) p = epsilon;
        else if (p > one_minus_epsilon) p = one_minus_epsilon;
        
        sum += t * logf(p);
    }
    
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        losses[i] = -sdata[0]; // Negate here (-sum is loss)
    }
}

__global__ void scale_loss_kernel(float* val, float s) {
    if (threadIdx.x == 0) *val *= s;
}

void categorical_cross_entropy_forward_cuda(
    const float* predictions,
    const float* targets,
    float* loss_output,
    int64_t batch_size,
    int64_t num_classes
) {
    if (batch_size == 0) return;
    
    float* d_losses;
    cudaMalloc(&d_losses, batch_size * sizeof(float));
    
    int threads = 256;
    int blocks = batch_size;
    
    // 1. Per-sample loss
    cce_forward_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions, targets, d_losses, batch_size, num_classes);
        
    // 2. Reduce sum
    int reduce_threads = 256;
    int reduce_blocks = (batch_size + reduce_threads - 1) / reduce_threads;
    
    if (reduce_blocks == 1) {
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, loss_output, batch_size);
    } else {
        float* d_partial;
        cudaMalloc(&d_partial, reduce_blocks * sizeof(float));
         sum_reduction_kernel<float><<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, d_partial, batch_size);
         sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_partial, loss_output, reduce_blocks);
        cudaFree(d_partial);
    }
    
    // 3. Average (Divide by batch_size)
    scale_loss_kernel<<<1, 1>>>(loss_output, 1.0f / static_cast<float>(batch_size));
    
    cudaFree(d_losses);
}

__global__ void cce_backward_kernel(
    const float* __restrict__ grad_output, // scalar
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ grad_input,
    int64_t numel,
    float scale // 1/N
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    const float epsilon = 1e-7f;
    const float one_minus_epsilon = 1.0f - 1e-7f;
    
    float p = predictions[i];
    float t = targets[i];
    float g = *grad_output; // Access scalar gradient (assumed on GPU)
    
    float grad = 0.0f;
    // d(log(clip(p))) = 1/clipped_p if not clamped, else 0
    if (p >= epsilon && p <= one_minus_epsilon) {
        grad = g * (-t / p);
    }
    
    grad_input[i] = grad * scale;
}

void categorical_cross_entropy_backward_cuda(
    const float* grad_output,
    const float* predictions,
    const float* targets,
    float* grad_input,
    int64_t batch_size,
    int64_t num_classes
) {
    int64_t numel = batch_size * num_classes;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    cce_backward_kernel<<<blocks, threads>>>(
        grad_output, predictions, targets, grad_input, numel, 1.0f / static_cast<float>(batch_size));
}


// ============================================================================
// MSE / MAE / BCE Extensions
// ============================================================================

// --- MSE ---
__global__ void mse_forward_kernel(
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ out,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float diff = p[i] - t[i];
    out[i] = diff * diff;
}

__global__ void mse_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ grad_in,
    int64_t n,
    float scale
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = *grad_out; // scalar
    float diff = p[i] - t[i];
    grad_in[i] = 2.0f * diff * g * scale;
}

void mse_loss_forward_cuda(const float* predictions, const float* targets, float* loss_output, int64_t numel) {
    if (numel == 0) return;
    float* d_losses;
    cudaMalloc(&d_losses, numel * sizeof(float));
    
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    mse_forward_kernel<<<blocks, threads>>>(predictions, targets, d_losses, numel);
    
    // Reduce
    int reduce_threads = 256;
    int reduce_blocks = (numel + reduce_threads - 1) / reduce_threads;
    
    if (reduce_blocks == 1) {
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, loss_output, numel);
    } else {
        float* d_partial;
        cudaMalloc(&d_partial, reduce_blocks * sizeof(float));
        sum_reduction_kernel<float><<<reduce_blocks, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, d_partial, numel);
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_partial, loss_output, reduce_blocks);
        cudaFree(d_partial);
    }
    
    scale_loss_kernel<<<1, 1>>>(loss_output, 1.0f / static_cast<float>(numel));
    cudaFree(d_losses);
}

void mse_loss_backward_cuda(const float* grad_output, const float* predictions, const float* targets, float* grad_input, int64_t numel) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    mse_backward_kernel<<<blocks, threads>>>(grad_output, predictions, targets, grad_input, numel, 1.0f/numel);
}

// --- MAE ---
__global__ void mae_forward_kernel(
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ out,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float diff = p[i] - t[i];
    out[i] = fabsf(diff);
}

__global__ void mae_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ grad_in,
    int64_t n,
    float scale
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = *grad_out;
    float diff = p[i] - t[i];
    float sign = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
    grad_in[i] = sign * g * scale;
}

void mae_loss_forward_cuda(const float* predictions, const float* targets, float* loss_output, int64_t numel) {
    if (numel == 0) return;
    float* d_losses;
    cudaMalloc(&d_losses, numel * sizeof(float));
    
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    mae_forward_kernel<<<blocks, threads>>>(predictions, targets, d_losses, numel);
    
    // Reduce (same logic as MSE, could abstract but copy-paste is safer for now without templating spaghetti)
    int reduce_threads = 256;
    int reduce_blocks = (numel + reduce_threads - 1) / reduce_threads;
    
    if (reduce_blocks == 1) {
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, loss_output, numel);
    } else {
        float* d_partial;
        cudaMalloc(&d_partial, reduce_blocks * sizeof(float));
        sum_reduction_kernel<float><<<reduce_blocks, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, d_partial, numel);
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_partial, loss_output, reduce_blocks);
        cudaFree(d_partial);
    }
    
    scale_loss_kernel<<<1, 1>>>(loss_output, 1.0f / static_cast<float>(numel));
    cudaFree(d_losses);
}

void mae_loss_backward_cuda(const float* grad_output, const float* predictions, const float* targets, float* grad_input, int64_t numel) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    mae_backward_kernel<<<blocks, threads>>>(grad_output, predictions, targets, grad_input, numel, 1.0f/numel);
}

// --- BCE ---
__global__ void bce_forward_kernel(
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ out,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float pi = p[i];
    float ti = t[i];
    
    // Clip
    const float eps = 1e-7f;
    const float one_minus_eps = 1.0f - 1e-7f;
    if (pi < eps) pi = eps;
    else if (pi > one_minus_eps) pi = one_minus_eps;
    
    // Loss = -(t * log(p) + (1-t) * log(1-p))
    out[i] = -(ti * logf(pi) + (1.0f - ti) * logf(1.0f - pi));
}

__global__ void bce_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ grad_in,
    int64_t n,
    float scale
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float g = *grad_out;
    float pi = p[i];
    float ti = t[i];
    
    // Clip
    const float eps = 1e-7f;
    const float one_minus_eps = 1.0f - eps;
    float p_clipped = pi;
    if (p_clipped < eps) p_clipped = eps;
    else if (p_clipped > one_minus_eps) p_clipped = one_minus_eps;
    
    // grad = (-t/p + (1-t)/(1-p)) * scale * g
    // if clipped, grad might be 0 theoretically, but standard DL frameworks usually pass gradient through clipped values or use logits.
    // Here we replicate strict derivative of the loss function formula with clipped p.
    
    float term1 = -ti / p_clipped;
    float term2 = (1.0f - ti) / (1.0f - p_clipped);
    grad_in[i] = (term1 + term2) * g * scale;
}

void bce_loss_forward_cuda(const float* predictions, const float* targets, float* loss_output, int64_t numel) {
    if (numel == 0) return;
    float* d_losses;
    cudaMalloc(&d_losses, numel * sizeof(float));
    
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    bce_forward_kernel<<<blocks, threads>>>(predictions, targets, d_losses, numel);
    
    // Reduce
    int reduce_threads = 256;
    int reduce_blocks = (numel + reduce_threads - 1) / reduce_threads;
    
    if (reduce_blocks == 1) {
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, loss_output, numel);
    } else {
        float* d_partial;
        cudaMalloc(&d_partial, reduce_blocks * sizeof(float));
        sum_reduction_kernel<float><<<reduce_blocks, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, d_partial, numel);
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_partial, loss_output, reduce_blocks);
        cudaFree(d_partial);
    }
    
    scale_loss_kernel<<<1, 1>>>(loss_output, 1.0f / static_cast<float>(numel));
    cudaFree(d_losses);
}

void bce_loss_backward_cuda(const float* grad_output, const float* predictions, const float* targets, float* grad_input, int64_t numel) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    bce_backward_kernel<<<blocks, threads>>>(grad_output, predictions, targets, grad_input, numel, 1.0f/numel);
}

} // namespace cuda
} // namespace OwnTensor