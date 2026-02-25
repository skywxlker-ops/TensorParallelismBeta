#include "ops/helpers/ActivationKernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace OwnTensor {
namespace cuda {

// Constants for GELU computation
__device__ constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
__device__ constexpr float GELU_COEF = 0.044715f;

// Fast tanh approximation (Direct hardware instruction via Inline PTX)
__device__ __forceinline__ float fast_tanh(float x) {
    float res;
    asm("tanh.approx.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
}

// Fast exponential approximation (exp(x) = 2^(x * log2(e)))
__device__ __forceinline__ float fast_exp(float x) {
    float res;
    float x_log2e = x * 1.44269504089f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(res) : "f"(x_log2e));
    return res;
}

// Fast reciprocal approximation (1/x)
__device__ __forceinline__ float fast_rcp(float x) {
    float res;
    asm("rcp.approx.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
}

// =============================================================================
// FUSED GELU KERNEL
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// =============================================================================

__global__ void fused_gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        float tanh_inner = fast_tanh(inner);
        output[i] = 0.5f * x * (1.0f + tanh_inner);
    }
}

// Vectorized version using float4 for better memory throughput
__global__ void fused_gelu_kernel_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time
    int64_t numel4 = numel / 4;
    for (int64_t i = idx; i < numel4; i += stride) {
        float4 x_vec = reinterpret_cast<const float4*>(input)[i];
        float4 out_vec;
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float x = (&x_vec.x)[j];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
            float tanh_inner = fast_tanh(inner);
            (&out_vec.x)[j] = 0.5f * x * (1.0f + tanh_inner);
        }
        
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }
    
    // Handle remaining elements
    for (int64_t i = numel4 * 4 + idx; i < numel; i += stride) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        float tanh_inner = fast_tanh(inner);
        output[i] = 0.5f * x * (1.0f + tanh_inner);
    }
}

void fused_gelu_cuda(
    const float* input,
    float* output,
    int64_t numel
) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    
    // Use vectorized kernel for large tensors where alignment is likely good
    if (numel >= 1024 && numel % 4 == 0) {
        int blocks4 = std::min((numel / 4 + threads - 1) / threads, (int64_t)65535);
        fused_gelu_kernel_vectorized<<<blocks4, threads>>>(input, output, numel);
    } else {
        fused_gelu_kernel<<<blocks, threads>>>(input, output, numel);
    }
}

// =============================================================================
// FUSED GELU BACKWARD KERNEL
// gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
// where u = sqrt(2/pi) * (x + 0.044715 * x^3)
// and du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
// =============================================================================

__global__ void fused_gelu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float grad = grad_output[i];
        
        float x2 = x * x;
        float x3 = x2 * x;
        
        float u = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        float du_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x2);
        
        float tanh_u = fast_tanh(u);
        float sech2_u = 1.0f - tanh_u * tanh_u;  // sech^2(u) = 1 - tanh^2(u)
        
        // gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
        float gelu_grad = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;
        
        grad_input[i] = grad * gelu_grad;
    }
}

void fused_gelu_backward_cuda(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int64_t numel
) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    fused_gelu_backward_kernel<<<blocks, threads>>>(grad_output, input, grad_input, numel);
}

// =============================================================================
// FUSED BIAS + GELU KERNEL
// output = gelu(input + bias)
// =============================================================================

__global__ void fused_bias_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t hidden_dim
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * hidden_dim;
    int64_t stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int64_t i = idx; i < total; i += stride) {
        int64_t bias_idx = i % hidden_dim;
        float x = input[i] + bias[bias_idx];
        
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        float tanh_inner = fast_tanh(inner);
        output[i] = 0.5f * x * (1.0f + tanh_inner);
    }
}

void fused_bias_gelu_cuda(
    const float* input,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t hidden_dim
) {
    int threads = 256;
    int64_t total = batch_size * hidden_dim;
    int blocks = std::min((total + threads - 1) / threads, (int64_t)65535);
    fused_bias_gelu_kernel<<<blocks, threads>>>(input, bias, output, batch_size, hidden_dim);
}


// =============================================================================
// RELU KERNELS
// =============================================================================

__global__ void relu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = val > 0.0f ? val : 0.0f;
    }
}

__global__ void relu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        float grad = grad_output[i];
        grad_input[i] = val > 0.0f ? grad : 0.0f;
    }
}

void relu_forward_cuda(const float* input, float* output, int64_t numel) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    relu_forward_kernel<<<blocks, threads>>>(input, output, numel);
}

void relu_backward_cuda(const float* grad_output, const float* input, float* grad_input, int64_t numel) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    relu_backward_kernel<<<blocks, threads>>>(grad_output, input, grad_input, numel);
}

// =============================================================================
// SIGMOID KERNELS
// =============================================================================

__global__ void sigmoid_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = fast_rcp(1.0f + fast_exp(-val));
    }
}

__global__ void sigmoid_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ output,
    float* __restrict__ grad_input,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        float s = output[i];
        float grad = grad_output[i];
        grad_input[i] = grad * s * (1.0f - s);
    }
}

void sigmoid_forward_cuda(const float* input, float* output, int64_t numel) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    sigmoid_forward_kernel<<<blocks, threads>>>(input, output, numel);
}

void sigmoid_backward_cuda(const float* grad_output, const float* output, float* grad_input, int64_t numel) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    sigmoid_backward_kernel<<<blocks, threads>>>(grad_output, output, grad_input, numel);
}

// =============================================================================
// SOFTMAX KERNELS
// =============================================================================

// Warp Reduce Helpers
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t rows,
    int64_t cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // One block per row
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // 1. Find Max for numerical stability
    float max_val = -INFINITY;
    #pragma unroll 4
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_input[i]);
    }
    
    // Block reduce max
    max_val = warpReduceMax(max_val);
    __shared__ float shared_max;
    if (threadIdx.x % warpSize == 0) {
        // Simple reduction across warps via atomic or shared mem
        // Since blockDim is usually 256 (8 warps), just use atomics for simplicity
        // But atomicMax for float needs int casting or specialized atomic
        // Let's use shared mem reduction for the warps
        // Placeholder for valid reduction logic
    }
    // Simplification: We need a robust block reduction.
    // For now, let's assume we use shared memory for partials
    
    // Single-warp reduction if cols is small? No, cols can be 50k.
    // Let's use standard atomicMax trick for float (assuming positive/sane floats? No, logits can be negatives)
    // Actually, let's use the standard "reduction in shared memory" approach.
    
    static __shared__ float s_max;
    if (threadIdx.x == 0) s_max = -INFINITY;
    __syncthreads();
    
    // Simple atomic Max for float (safe for standard floats)
    // atomicMax only supports int/uint. 
    // We can cast to int if valid, but negative floats order backwards as ints.
    // Let's rely on thread 0 doing final reduction of warps if block is small.
    // But let's write a proper loop.
    
    // Manual block reduce:
    // Each warp puts result in shared mem
    static __shared__ float warp_vals[32]; // Max 1024 threads = 32 warps
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    
    if (laneId == 0) warp_vals[warpId] = -INFINITY;
    __syncthreads();
    
    if (laneId == 0) warp_vals[warpId] = max_val;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float block_max = -INFINITY;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i=0; i<num_warps; ++i) {
            block_max = fmaxf(block_max, warp_vals[i]);
        }
        s_max = block_max;
    }
    __syncthreads();
    max_val = s_max;
    
    // 2. Compute Exp and Sum
    float sum_exp = 0.0f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = expf(row_input[i] - max_val);
        // Store temporarily in output to avoid recomputing exp (optional, but saves ALUS)
        // Actually better to just write it if we have memory bandwidth
        // But we need to divide later.
        // Let's just compute sum.
        sum_exp += val;
        row_output[i] = val; // Store exp(x-max)
    }
    
    sum_exp = warpReduceSum(sum_exp);
    
    // Block reduce sum
    static __shared__ float s_sum;
    if (laneId == 0) warp_vals[warpId] = sum_exp;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i=0; i<num_warps; ++i) {
            block_sum += warp_vals[i];
        }
        s_sum = block_sum;
    }
    __syncthreads();
    sum_exp = s_sum;
    
    // 3. Normalize
    float inv_sum = fast_rcp(sum_exp);
    #pragma unroll 4
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_output[i] *= inv_sum;
    }
}

__global__ void softmax_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ output,
    float* __restrict__ grad_input,
    int64_t rows,
    int64_t cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_grad = grad_output + row * cols;
    const float* row_out = output + row * cols;
    float* row_gin = grad_input + row * cols;
    
    // 1. Compute dot product sum(grad * out)
    float dot = 0.0f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        dot += row_grad[i] * row_out[i];
    }
    
    dot = warpReduceSum(dot);
    
    static __shared__ float s_dot;
    static __shared__ float warp_vals[32];
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    
    if (laneId == 0) warp_vals[warpId] = dot;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float block_dot = 0.0f;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i=0; i<num_warps; ++i) {
            block_dot += warp_vals[i];
        }
        s_dot = block_dot;
    }
    __syncthreads();
    dot = s_dot;
    
    // 2. Compute grad_input = out * (grad - dot)
    #pragma unroll 4
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float s = row_out[i];
        float g = row_grad[i];
        row_gin[i] = s * (g - dot);
    }
}

void softmax_forward_cuda(const float* input, float* output, int64_t rows, int64_t cols) {
    int threads = (cols <= 1024) ? 256 : 1024;
    // Ensure threads is multiple of 32 for warp ops
    if (threads < 32) threads = 32;
    
    // One block per row
    dim3 blocks(rows);
    softmax_forward_kernel<<<blocks, threads>>>(input, output, rows, cols);
}

void softmax_backward_cuda(const float* grad_output, const float* output, float* grad_input, int64_t rows, int64_t cols) {
    int threads = (cols <= 1024) ? 256 : 1024;
    if (threads < 32) threads = 32;
    
    dim3 blocks(rows);
    softmax_backward_kernel<<<blocks, threads>>>(grad_output, output, grad_input, rows, cols);
}

} // namespace cuda
} // namespace OwnTensor

