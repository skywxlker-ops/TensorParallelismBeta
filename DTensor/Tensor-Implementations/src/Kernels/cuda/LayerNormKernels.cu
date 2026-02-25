#include "ops/helpers/LayerNormKernels.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// Unused headers removed

namespace OwnTensor {
namespace cuda {

// =================================================================================
// Helper: Warp Reduction
// =================================================================================
template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =================================================================================
// Forward Kernel
// =================================================================================
// Grid: [rows], Block: [min(cols, 1024)] (or fixed size with loop)
// We'll use 1 block per row.
template<typename T, typename AccT>
__global__ void layer_norm_forward_kernel(
    const T* __restrict__ x,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ y,
    AccT* __restrict__ mean_out,
    AccT* __restrict__ rstd_out,
    int cols,
    AccT eps) 
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Offset to current row
    const T* row_x = x + row * cols;
    T* row_y = y + row * cols;

    AccT sum = (AccT)0;
    #pragma unroll 4
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += (AccT)row_x[i];
    }
    sum = warpReduceSum<AccT>(sum);
    // Block reduction
    __shared__ AccT shared_val;
    if (tid == 0) shared_val = (AccT)0;
    __syncthreads();
    
    if (tid % warpSize == 0) atomicAdd(&shared_val, sum);
    __syncthreads();
    
    AccT mu = shared_val / cols;
    if (tid == 0) mean_out[row] = mu;

    AccT sum_sq = (AccT)0;
    #pragma unroll 4
    for (int i = tid; i < cols; i += blockDim.x) {
        AccT diff = (AccT)row_x[i] - mu;
        sum_sq += diff * diff;
    }
    sum_sq = warpReduceSum<AccT>(sum_sq);
    
    if (tid == 0) shared_val = (AccT)0; // Reuse shared
    __syncthreads();
    
    if (tid % warpSize == 0) atomicAdd(&shared_val, sum_sq);
    __syncthreads();
    
    AccT var = shared_val / cols;
    AccT rstd = rsqrtf(var + eps);
    if (tid == 0) rstd_out[row] = rstd;

    // 3. Normalize and Output
    #pragma unroll 4
    for (int i = tid; i < cols; i += blockDim.x) {
        AccT val = ((AccT)row_x[i] - mu) * rstd;
        
        AccT g = (gamma) ? (AccT)gamma[i] : (AccT)1.0f;
        AccT b = (beta) ? (AccT)beta[i] : (AccT)0.0f;
        
        row_y[i] = (T)(val * g + b);
    }
}


void layer_norm_forward_cuda(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    float* mean,
    float* rstd,
    int rows,
    int cols,
    float eps)
{
    int threads = 256;
    layer_norm_forward_kernel<float, float><<<rows, threads>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}

void layer_norm_forward_cuda(
    const __half* x,
    const __half* gamma,
    const __half* beta,
    __half* y,
    float* mean,
    float* rstd,
    int rows,
    int cols,
    float eps)
{
    int threads = 256;
    layer_norm_forward_kernel<__half, float><<<rows, threads>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}

void layer_norm_forward_cuda(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta,
    __nv_bfloat16* y,
    float* mean,
    float* rstd,
    int rows,
    int cols,
    float eps)
{
    int threads = 256;
    layer_norm_forward_kernel<__nv_bfloat16, float><<<rows, threads>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}

// =================================================================================
// Backward Kernels
// =================================================================================

// Kernel 1: Compute gradients for Gamma and Beta (Reduce over Rows)
// Grid: [cols], Block: [256]
// Each block handles one column (feature), reduces over all rows.
// Very simple implementation, might be slow for massive batch sizes but fine for GPT-2.
// __global__ void ln_backward_gamma_beta_kernel(
//     const float* __restrict__ grad_y,
//     const float* __restrict__ x,
//     const float* __restrict__ mean,
//     const float* __restrict__ rstd,
//     float* __restrict__ grad_gamma,
//     float* __restrict__ grad_beta,
//     int rows,
//     int cols)
// {
//     // Block handles 32 columns, 8 rows of threads (Total 256 threads)
//     int tx = threadIdx.x; // Column index within tile (0-31)
//     int ty = threadIdx.y; // Row index within tile (0-7)
    
//     // Shared memory for reduction across the 8 rows in the block
//     __shared__ float s_dgamma[8][32];
//     __shared__ float s_dbeta[8][32];
//     for (int col_base = blockIdx.x * 32; col_base < cols; col_base += gridDim.x * 32) {
//         int col = col_base + tx;
        
//         float d_gamma_acc = 0.0f;
//         float d_beta_acc = 0.0f;
//         // Cooperative row processing
//         if (col < cols) {
//             for (int row = ty; row < rows; row += 8) { // 8 is blockDim.y
//                 float gy = grad_y[row * cols + col]; // COALESCED!
//                 float input_val = x[row * cols + col]; // COALESCED!
//                 float m = mean[row];
//                 float rs = rstd[row];
                
//                 float norm_x = (input_val - m) * rs;
//                 d_beta_acc += gy;
//                 d_gamma_acc += gy * norm_x;
//             }
//         }
//         // Store partial sums in shared memory
//         s_dgamma[ty][tx] = d_gamma_acc;
//         s_dbeta[ty][tx] = d_beta_acc;
//         __syncthreads();
//         // Reduce across the 8 threads that handled different rows for this column
//         if (ty == 0 && col < cols) {
//             float final_dgamma = 0, final_dbeta = 0;
//             #pragma unroll
//             for (int i = 0; i < 8; i++) {
//                 final_dgamma += s_dgamma[i][tx];
//                 final_dbeta += s_dbeta[i][tx];
//             }
            
//             // Atomic add to global memory (each block handles a different set of rows/cols)
//             // If grid_stride over rows was applied, we'd need atomicAdd.
//             // Since we iterate over ALL rows in this loop, we can just write if we use block stride over cols.
//             // However, to be safe and support multiple blocks per col range:
//             atomicAdd(&grad_gamma[col], final_dgamma);
//             atomicAdd(&grad_beta[col], final_dbeta);
//         }
//         __syncthreads();
//     }
// }
__global__ void ln_backward_gamma_beta_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int rows,
    int cols)
{
    int tx = threadIdx.x; // Column offset (0-31)
    int ty = threadIdx.y; // Row offset (0-7)
    
    __shared__ float s_dgamma[8][32];
    __shared__ float s_dbeta[8][32];

    // 2D Grid Stride Loop: 
    // blockIdx.x handles columns, blockIdx.y handles rows
    #pragma unroll 4
    for (int col_base = blockIdx.x * 32; col_base < cols; col_base += gridDim.x * 32) {
        int col = col_base + tx;
        
        float d_gamma_acc = 0.0f;
        float d_beta_acc = 0.0f;

        if (col < cols) {
            // STRIDE OVER ROWS using gridDim.y
            // Each block in the Y-dimension handles a subset of rows
            for (int row = blockIdx.y * 8 + ty; row < rows; row += gridDim.y * 8) {
                float gy = grad_y[row * cols + col];
                float input_val = x[row * cols + col];
                float m = mean[row];
                float rs = rstd[row];
                
                float norm_x = (input_val - m) * rs;
                d_beta_acc += gy;
                d_gamma_acc += gy * norm_x;
            }
        }

        s_dgamma[ty][tx] = d_gamma_acc;
        s_dbeta[ty][tx] = d_beta_acc;
        __syncthreads();

        if (ty == 0 && col < cols) {
            float final_dgamma = 0, final_dbeta = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                final_dgamma += s_dgamma[i][tx];
                final_dbeta += s_dbeta[i][tx];
            }
            // MUST use atomicAdd as multiple blocks (gridDim.y) 
            // are now contributing to the same grad_gamma[col]
            atomicAdd(&grad_gamma[col], final_dgamma);
            atomicAdd(&grad_beta[col], final_dbeta);
        }
        __syncthreads();
    }
}

// Kernel 2: Compute Gradients for Input (Per Row)
// Standard derivation for LayerNorm backward
__global__ void ln_backward_input_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ gamma,
    float* __restrict__ grad_x,
    int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* dy_row = grad_y + row * cols;
    const float* x_row = x + row * cols;
    float* dx_row = grad_x + row * cols;
    
    float m = mean[row];
    float rs = rstd[row];
    
    // 1. Compute local generic reductions: sum(dy * gamma) and sum(dy * gamma * (x-m))
    float sum_dy_gamma = 0.0f;
    float sum_dy_gamma_norm = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = (gamma) ? gamma[i] : 1.0f;
        float dy = dy_row[i];
        float val = x_row[i];
        float norm_x = (val - m) * rs;
        
        sum_dy_gamma += dy * g;
        sum_dy_gamma_norm += dy * g * norm_x;
    }
    
    sum_dy_gamma = warpReduceSum(sum_dy_gamma);
    sum_dy_gamma_norm = warpReduceSum(sum_dy_gamma_norm);
    
    __shared__ float s_sum1, s_sum2;
    if (tid == 0) { s_sum1 = 0; s_sum2 = 0; }
    __syncthreads();
    
    if (tid % warpSize == 0) {
        atomicAdd(&s_sum1, sum_dy_gamma);
        atomicAdd(&s_sum2, sum_dy_gamma_norm);
    }
    __syncthreads();
    
    float total_sum1 = s_sum1;
    float total_sum2 = s_sum2;
    
    // 2. Compute dx
    // dxhat = (dy * gamma)
    // dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
    //    = rstd * (dy*gamma - (1/D)*sum(dy*gamma) - xhat * (1/D)*sum(dy*gamma*xhat))
    float inv_cols = 1.0f / cols;
    
    #pragma unroll 4
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = (gamma) ? gamma[i] : 1.0f;
        float dy = dy_row[i];
        float val = x_row[i];
        float norm_x = (val - m) * rs;
        
        float term1 = dy * g;
        float term2 = total_sum1; 
        float term3 = norm_x * total_sum2;
        
        dx_row[i] = rs * (term1 - (term2 + term3) * inv_cols);
    }
}


void layer_norm_backward_cuda(
    const float* grad_y,
    const float* x,
    const float* mean,
    const float* rstd,
    const float* gamma,
    float* grad_x,
    float* grad_gamma,
    float* grad_beta,
    int rows,
    int cols)
{
    // 1. Gradients for Weights (Gamma/Beta)
    if (grad_gamma != nullptr || grad_beta != nullptr) {
        cudaMemset(grad_gamma, 0, cols * sizeof(float));
        cudaMemset(grad_beta, 0, cols * sizeof(float));

        dim3 threads(32, 8); // 256 threads per block
        
        // X handles Columns
        int blocks_x = (cols + 31) / 32;
        
        // Y handles Rows - aim for ~160-320 total blocks to saturate SMs
        // If blocks_x is 24, we can set blocks_y to 10 or 12.
        int blocks_y = 128 / blocks_x; 
        if (blocks_y < 1) blocks_y = 1;
        if (blocks_y > 32) blocks_y = 32; // Limit to avoid too much atomic contention

        dim3 grid(blocks_x, blocks_y);

        ln_backward_gamma_beta_kernel<<<grid, threads>>>(
            grad_y, x, mean, rstd, grad_gamma, grad_beta, rows, cols
        );
    }
    
    // 2. Gradients for Input
    if (grad_x != nullptr) {
        int threads = 256;
        if (cols > 256) threads = 512;
        ln_backward_input_kernel<<<rows, threads>>>(
            grad_y, x, mean, rstd, gamma, grad_x, cols
        );
    }
}

} // namespace cuda
} // namespace OwnTensor
