#include "dnn/VectorizedLayerNormKernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace OwnTensor {
namespace cuda {

// =============================================================================
// Warp + block reduction helpers
// =============================================================================

__device__ __forceinline__ float vln_warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Reduce a per-thread float across the whole block.
// Uses smem[32] for warp leaders; result broadcast in smem[0].
// Caller must __syncthreads() before reading smem[0].
__device__ __forceinline__ float vln_block_reduce(float val, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    val = vln_warp_reduce(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();

    // Only first warp sums the warp results
    float s = 0.0f;
    if (threadIdx.x == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nwarps; w++) s += smem[w];
        smem[0] = s;
    }
    __syncthreads();
    return smem[0];
}

// =============================================================================
// float32 forward kernel
//   Vectorized path : cols % 4 == 0  → float4 loads/stores
//   Scalar  fallback: all other cases
// =============================================================================
__global__ void vln_fwd_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int cols, float eps, bool vectorized)
{
    extern __shared__ float smem[];   // 32 floats for block reduce

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    const float* xr = x + (int64_t)row * cols;
    float* yr = y + (int64_t)row * cols;

    // ----------------------------------------------------------------
    // Step 1: mean
    // ----------------------------------------------------------------
    float sum = 0.0f;
    if (vectorized) {
        const float4* xv = reinterpret_cast<const float4*>(xr);
        int vcols = cols >> 2;
        for (int i = tid; i < vcols; i += bdim) {
            float4 v = xv[i];
            sum += v.x + v.y + v.z + v.w;
        }
    } else {
        for (int i = tid; i < cols; i += bdim)
            sum += xr[i];
    }
    float mu = vln_block_reduce(sum, smem) / cols;
    if (tid == 0) mean_out[row] = mu;

    // ----------------------------------------------------------------
    // Step 2: variance
    // ----------------------------------------------------------------
    float var = 0.0f;
    if (vectorized) {
        const float4* xv = reinterpret_cast<const float4*>(xr);
        int vcols = cols >> 2;
        for (int i = tid; i < vcols; i += bdim) {
            float4 v = xv[i];
            float d0 = v.x - mu, d1 = v.y - mu, d2 = v.z - mu, d3 = v.w - mu;
            var += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float d = xr[i] - mu;
            var += d * d;
        }
    }
    float rstd = rsqrtf(vln_block_reduce(var, smem) / cols + eps);
    if (tid == 0) rstd_out[row] = rstd;
    // Broadcast rstd to all threads via smem
    if (tid == 0) smem[0] = rstd;
    __syncthreads();
    rstd = smem[0];

    // ----------------------------------------------------------------
    // Step 3: normalize and output
    // ----------------------------------------------------------------
    if (vectorized) {
        const float4* xv  = reinterpret_cast<const float4*>(xr);
        const float4* gv  = gamma ? reinterpret_cast<const float4*>(gamma) : nullptr;
        const float4* bv  = beta  ? reinterpret_cast<const float4*>(beta)  : nullptr;
        float4* yv = reinterpret_cast<float4*>(yr);
        int vcols = cols >> 2;
        for (int i = tid; i < vcols; i += bdim) {
            float4 v = xv[i];
            float4 g = gv ? gv[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 b = bv ? bv[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 out;
            out.x = ((v.x - mu) * rstd) * g.x + b.x;
            out.y = ((v.y - mu) * rstd) * g.y + b.y;
            out.z = ((v.z - mu) * rstd) * g.z + b.z;
            out.w = ((v.w - mu) * rstd) * g.w + b.w;
            yv[i] = out;
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float g = gamma ? gamma[i] : 1.0f;
            float b = beta  ? beta[i]  : 0.0f;
            yr[i] = ((xr[i] - mu) * rstd) * g + b;
        }
    }
}

// =============================================================================
// float32 backward kernel: input gradients (per row)
// =============================================================================
__global__ void vln_bwd_input_f32_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ gamma,
    float* __restrict__ dx,
    int cols, bool vectorized)
{
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    const float* dy_row = dy + (int64_t)row * cols;
    const float* x_row  = x  + (int64_t)row * cols;
    float*       dx_row = dx + (int64_t)row * cols;
    float mu   = mean[row];
    float rs   = rstd[row];

    float sum1 = 0.0f, sum2 = 0.0f;

    if (vectorized) {
        const float4* dv = reinterpret_cast<const float4*>(dy_row);
        const float4* xv = reinterpret_cast<const float4*>(x_row);
        const float4* gv = gamma ? reinterpret_cast<const float4*>(gamma) : nullptr;
        int vcols = cols >> 2;
        for (int i = tid; i < vcols; i += bdim) {
            float4 d = dv[i];
            float4 xx = xv[i];
            float4 g = gv ? gv[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float n0 = (xx.x - mu) * rs, n1 = (xx.y - mu) * rs;
            float n2 = (xx.z - mu) * rs, n3 = (xx.w - mu) * rs;
            sum1 += d.x*g.x + d.y*g.y + d.z*g.z + d.w*g.w;
            sum2 += d.x*g.x*n0 + d.y*g.y*n1 + d.z*g.z*n2 + d.w*g.w*n3;
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float g = gamma ? gamma[i] : 1.0f;
            float d = dy_row[i];
            float n = (x_row[i] - mu) * rs;
            sum1 += d * g;
            sum2 += d * g * n;
        }
    }

    float total1 = vln_block_reduce(sum1, smem);
    // Save total1 beyond the 32 warp-leader slots (smem[0..31]) so the second
    // block_reduce does not overwrite it.
    if (tid == 0) smem[32] = total1;
    __syncthreads();
    float total2 = vln_block_reduce(sum2, smem);
    total1 = smem[32];
    float inv_cols = 1.0f / cols;

    if (vectorized) {
        const float4* dv = reinterpret_cast<const float4*>(dy_row);
        const float4* xv = reinterpret_cast<const float4*>(x_row);
        const float4* gv = gamma ? reinterpret_cast<const float4*>(gamma) : nullptr;
        float4* dxv = reinterpret_cast<float4*>(dx_row);
        int vcols = cols >> 2;
        for (int i = tid; i < vcols; i += bdim) {
            float4 d = dv[i];
            float4 xx = xv[i];
            float4 g = gv ? gv[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 out;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float dj  = (&d.x)[j];
                float xj  = (&xx.x)[j];
                float gj  = (&g.x)[j];
                float nxj = (xj - mu) * rs;
                (&out.x)[j] = rs * (dj*gj - (total1 + nxj*total2) * inv_cols);
            }
            dxv[i] = out;
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float g  = gamma ? gamma[i] : 1.0f;
            float d  = dy_row[i];
            float nx = (x_row[i] - mu) * rs;
            dx_row[i] = rs * (d*g - (total1 + nx*total2) * inv_cols);
        }
    }
}

// =============================================================================
// float32 backward kernel: gamma/beta gradients (reduce over rows)
// 2D grid: blockIdx.x → column tile (32 wide), blockIdx.y → row tile
// Uses float4 loads for better memory throughput; atomicAdd for cross-block.
// =============================================================================
__global__ void vln_bwd_gamma_beta_f32_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows, int cols)
{
    int tx = threadIdx.x;  // 0..31 (column within tile)
    int ty = threadIdx.y;  // 0..7  (row within tile)

    __shared__ float s_dg[8][32];
    __shared__ float s_db[8][32];

    for (int col_base = blockIdx.x * 32; col_base < cols; col_base += gridDim.x * 32) {
        int col = col_base + tx;
        float dg_acc = 0.0f, db_acc = 0.0f;

        if (col < cols) {
            for (int row = blockIdx.y * 8 + ty; row < rows; row += gridDim.y * 8) {
                float gy   = dy[row * cols + col];
                float xval = x[row * cols + col];
                float nx   = (xval - mean[row]) * rstd[row];
                db_acc += gy;
                dg_acc += gy * nx;
            }
        }
        s_dg[ty][tx] = dg_acc;
        s_db[ty][tx] = db_acc;
        __syncthreads();

        if (ty == 0 && col < cols) {
            float fg = 0.0f, fb = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) { fg += s_dg[i][tx]; fb += s_db[i][tx]; }
            if (dgamma) atomicAdd(&dgamma[col], fg);
            if (dbeta)  atomicAdd(&dbeta[col],  fb);
        }
        __syncthreads();
    }
}

// =============================================================================
// Launchers: float32
// =============================================================================

void vln_forward_f32(
    const float* x, const float* gamma, const float* beta,
    float* y, float* mean_out, float* rstd_out,
    int rows, int cols, float eps)
{
    bool vec = (cols % 4 == 0);
    int threads = std::min(std::max(cols / (vec ? 4 : 1), 32), 256);
    // Align to warp size
    threads = ((threads + 31) / 32) * 32;
    // smem: 32 floats for block reduce
    size_t smem = 32 * sizeof(float);
    vln_fwd_f32_kernel<<<rows, threads, smem>>>(
        x, gamma, beta, y, mean_out, rstd_out, cols, eps, vec);
}

void vln_backward_f32(
    const float* dy, const float* x,
    const float* mean, const float* rstd, const float* gamma,
    float* dx, float* dgamma, float* dbeta, int rows, int cols)
{
    bool vec = (cols % 4 == 0);

    if (dgamma || dbeta) {
        if (dgamma) cudaMemset(dgamma, 0, cols * sizeof(float));
        if (dbeta)  cudaMemset(dbeta,  0, cols * sizeof(float));
        dim3 threads(32, 8);
        int bx = (cols + 31) / 32;
        int by = std::max(1, std::min(32, 128 / bx));
        vln_bwd_gamma_beta_f32_kernel<<<dim3(bx, by), threads>>>(
            dy, x, mean, rstd, dgamma, dbeta, rows, cols);
    }

    if (dx) {
        int threads = std::min(std::max(cols / (vec ? 4 : 1), 32), 512);
        threads = ((threads + 31) / 32) * 32;
        size_t smem = 34 * sizeof(float);  // 32 for block_reduce + 2 scratch slots
        vln_bwd_input_f32_kernel<<<rows, threads, smem>>>(
            dy, x, mean, rstd, gamma, dx, cols, vec);
    }
}

// =============================================================================
// fp16 helpers
// =============================================================================

__device__ __forceinline__ float2 half2_to_float2(__half2 h) {
    return __half22float2(h);
}
__device__ __forceinline__ __half2 float2_to_half2(float a, float b) {
    return __floats2half2_rn(a, b);
}

// =============================================================================
// fp16 forward kernel
// =============================================================================
__global__ void vln_fwd_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    __half* __restrict__ y,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int cols, float eps, bool vectorized)
{
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    const __half* xr = x + (int64_t)row * cols;
    __half* yr = y + (int64_t)row * cols;

    float sum = 0.0f;
    if (vectorized) {
        const __half2* xv = reinterpret_cast<const __half2*>(xr);
        int hcols = cols >> 1;
        for (int i = tid; i < hcols; i += bdim) {
            float2 v = half2_to_float2(xv[i]);
            sum += v.x + v.y;
        }
    } else {
        for (int i = tid; i < cols; i += bdim)
            sum += __half2float(xr[i]);
    }
    float mu = vln_block_reduce(sum, smem) / cols;
    if (tid == 0) mean_out[row] = mu;

    float var = 0.0f;
    if (vectorized) {
        const __half2* xv = reinterpret_cast<const __half2*>(xr);
        int hcols = cols >> 1;
        for (int i = tid; i < hcols; i += bdim) {
            float2 v = half2_to_float2(xv[i]);
            float d0 = v.x - mu, d1 = v.y - mu;
            var += d0*d0 + d1*d1;
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float d = __half2float(xr[i]) - mu; var += d*d;
        }
    }
    float rstd = rsqrtf(vln_block_reduce(var, smem) / cols + eps);
    if (tid == 0) rstd_out[row] = rstd;
    if (tid == 0) smem[0] = rstd;
    __syncthreads();
    rstd = smem[0];

    if (vectorized) {
        const __half2* xv = reinterpret_cast<const __half2*>(xr);
        const __half2* gv = gamma ? reinterpret_cast<const __half2*>(gamma) : nullptr;
        const __half2* bv = beta  ? reinterpret_cast<const __half2*>(beta)  : nullptr;
        __half2* yv = reinterpret_cast<__half2*>(yr);
        int hcols = cols >> 1;
        for (int i = tid; i < hcols; i += bdim) {
            float2 v = half2_to_float2(xv[i]);
            float2 g = gv ? half2_to_float2(gv[i]) : make_float2(1.0f, 1.0f);
            float2 b = bv ? half2_to_float2(bv[i]) : make_float2(0.0f, 0.0f);
            float o0 = ((v.x - mu) * rstd) * g.x + b.x;
            float o1 = ((v.y - mu) * rstd) * g.y + b.y;
            yv[i] = float2_to_half2(o0, o1);
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float g = gamma ? __half2float(gamma[i]) : 1.0f;
            float b = beta  ? __half2float(beta[i])  : 0.0f;
            yr[i] = __float2half(
                ((__half2float(xr[i]) - mu) * rstd) * g + b);
        }
    }
}

// =============================================================================
// fp16 backward kernel: input gradients
// =============================================================================
__global__ void vln_bwd_input_f16_kernel(
    const __half* __restrict__ dy,
    const __half* __restrict__ x,
    const float*  __restrict__ mean,
    const float*  __restrict__ rstd,
    const __half* __restrict__ gamma,
    __half* __restrict__ dx,
    int cols, bool vectorized)
{
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    const __half* dy_row = dy + (int64_t)row * cols;
    const __half* x_row  = x  + (int64_t)row * cols;
    __half*       dx_row = dx + (int64_t)row * cols;
    float mu = mean[row], rs = rstd[row];

    float sum1 = 0.0f, sum2 = 0.0f;
    if (vectorized) {
        const __half2* dv = reinterpret_cast<const __half2*>(dy_row);
        const __half2* xv = reinterpret_cast<const __half2*>(x_row);
        const __half2* gv = gamma ? reinterpret_cast<const __half2*>(gamma) : nullptr;
        int hcols = cols >> 1;
        for (int i = tid; i < hcols; i += bdim) {
            float2 d = half2_to_float2(dv[i]);
            float2 xx = half2_to_float2(xv[i]);
            float2 g = gv ? half2_to_float2(gv[i]) : make_float2(1.0f, 1.0f);
            float n0 = (xx.x - mu) * rs, n1 = (xx.y - mu) * rs;
            sum1 += d.x*g.x + d.y*g.y;
            sum2 += d.x*g.x*n0 + d.y*g.y*n1;
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float g = gamma ? __half2float(gamma[i]) : 1.0f;
            float d = __half2float(dy_row[i]);
            float n = (__half2float(x_row[i]) - mu) * rs;
            sum1 += d * g; sum2 += d * g * n;
        }
    }

    float total1 = vln_block_reduce(sum1, smem);
    if (tid == 0) smem[32] = total1;
    __syncthreads();
    float total2 = vln_block_reduce(sum2, smem);
    total1 = smem[32];
    float inv_cols = 1.0f / cols;

    if (vectorized) {
        const __half2* dv = reinterpret_cast<const __half2*>(dy_row);
        const __half2* xv = reinterpret_cast<const __half2*>(x_row);
        const __half2* gv = gamma ? reinterpret_cast<const __half2*>(gamma) : nullptr;
        __half2* dxv = reinterpret_cast<__half2*>(dx_row);
        int hcols = cols >> 1;
        for (int i = tid; i < hcols; i += bdim) {
            float2 d = half2_to_float2(dv[i]);
            float2 xx = half2_to_float2(xv[i]);
            float2 g = gv ? half2_to_float2(gv[i]) : make_float2(1.0f, 1.0f);
            float n0 = (xx.x - mu) * rs, n1 = (xx.y - mu) * rs;
            float o0 = rs * (d.x*g.x - (total1 + n0*total2) * inv_cols);
            float o1 = rs * (d.y*g.y - (total1 + n1*total2) * inv_cols);
            dxv[i] = float2_to_half2(o0, o1);
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float g  = gamma ? __half2float(gamma[i]) : 1.0f;
            float d  = __half2float(dy_row[i]);
            float nx = (__half2float(x_row[i]) - mu) * rs;
            dx_row[i] = __float2half(rs * (d*g - (total1 + nx*total2) * inv_cols));
        }
    }
}

// =============================================================================
// fp16 backward kernel: gamma/beta gradients (float32 output)
// =============================================================================
__global__ void vln_bwd_gamma_beta_f16_kernel(
    const __half* __restrict__ dy,
    const __half* __restrict__ x,
    const float*  __restrict__ mean,
    const float*  __restrict__ rstd,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows, int cols)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float s_dg[8][32];
    __shared__ float s_db[8][32];

    for (int col_base = blockIdx.x * 32; col_base < cols; col_base += gridDim.x * 32) {
        int col = col_base + tx;
        float dg_acc = 0.0f, db_acc = 0.0f;
        if (col < cols) {
            for (int row = blockIdx.y * 8 + ty; row < rows; row += gridDim.y * 8) {
                float gy   = __half2float(dy[row * cols + col]);
                float xval = __half2float(x[row * cols + col]);
                float nx   = (xval - mean[row]) * rstd[row];
                db_acc += gy; dg_acc += gy * nx;
            }
        }
        s_dg[ty][tx] = dg_acc; s_db[ty][tx] = db_acc;
        __syncthreads();
        if (ty == 0 && col < cols) {
            float fg = 0.0f, fb = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) { fg += s_dg[i][tx]; fb += s_db[i][tx]; }
            if (dgamma) atomicAdd(&dgamma[col], fg);
            if (dbeta)  atomicAdd(&dbeta[col],  fb);
        }
        __syncthreads();
    }
}

// =============================================================================
// Launchers: fp16
// =============================================================================

void vln_forward_f16(
    const __half* x, const __half* gamma, const __half* beta,
    __half* y, float* mean_out, float* rstd_out, int rows, int cols, float eps)
{
    bool vec = (cols % 2 == 0);
    int threads = std::min(std::max(cols / (vec ? 2 : 1), 32), 256);
    threads = ((threads + 31) / 32) * 32;
    size_t smem = 32 * sizeof(float);
    vln_fwd_f16_kernel<<<rows, threads, smem>>>(
        x, gamma, beta, y, mean_out, rstd_out, cols, eps, vec);
}

void vln_backward_f16(
    const __half* dy, const __half* x,
    const float* mean, const float* rstd, const __half* gamma,
    __half* dx, float* dgamma, float* dbeta, int rows, int cols)
{
    bool vec = (cols % 2 == 0);

    if (dgamma || dbeta) {
        if (dgamma) cudaMemset(dgamma, 0, cols * sizeof(float));
        if (dbeta)  cudaMemset(dbeta,  0, cols * sizeof(float));
        dim3 threads(32, 8);
        int bx = (cols + 31) / 32;
        int by = std::max(1, std::min(32, 128 / bx));
        vln_bwd_gamma_beta_f16_kernel<<<dim3(bx, by), threads>>>(
            dy, x, mean, rstd, dgamma, dbeta, rows, cols);
    }

    if (dx) {
        int threads = std::min(std::max(cols / (vec ? 2 : 1), 32), 512);
        threads = ((threads + 31) / 32) * 32;
        size_t smem = 34 * sizeof(float);
        vln_bwd_input_f16_kernel<<<rows, threads, smem>>>(
            dy, x, mean, rstd, gamma, dx, cols, vec);
    }
}

// =============================================================================
// bf16 helpers
// =============================================================================

__device__ __forceinline__ float2 bfloat162_to_float2(__nv_bfloat162 h) {
    return __bfloat1622float2(h);
}
__device__ __forceinline__ __nv_bfloat162 float2_to_bfloat162(float a, float b) {
    return __floats2bfloat162_rn(a, b);
}

// =============================================================================
// bf16 forward kernel
// =============================================================================
__global__ void vln_fwd_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ y,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int cols, float eps, bool vectorized)
{
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    const __nv_bfloat16* xr = x + (int64_t)row * cols;
    __nv_bfloat16* yr = y + (int64_t)row * cols;

    float sum = 0.0f;
    if (vectorized) {
        const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(xr);
        int hcols = cols >> 1;
        for (int i = tid; i < hcols; i += bdim) {
            float2 v = bfloat162_to_float2(xv[i]);
            sum += v.x + v.y;
        }
    } else {
        for (int i = tid; i < cols; i += bdim)
            sum += __bfloat162float(xr[i]);
    }
    float mu = vln_block_reduce(sum, smem) / cols;
    if (tid == 0) mean_out[row] = mu;

    float var = 0.0f;
    if (vectorized) {
        const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(xr);
        int hcols = cols >> 1;
        for (int i = tid; i < hcols; i += bdim) {
            float2 v = bfloat162_to_float2(xv[i]);
            float d0 = v.x - mu, d1 = v.y - mu;
            var += d0*d0 + d1*d1;
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float d = __bfloat162float(xr[i]) - mu; var += d*d;
        }
    }
    float rstd = rsqrtf(vln_block_reduce(var, smem) / cols + eps);
    if (tid == 0) rstd_out[row] = rstd;
    if (tid == 0) smem[0] = rstd;
    __syncthreads();
    rstd = smem[0];

    if (vectorized) {
        const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(xr);
        const __nv_bfloat162* gv = gamma ? reinterpret_cast<const __nv_bfloat162*>(gamma) : nullptr;
        const __nv_bfloat162* bv = beta  ? reinterpret_cast<const __nv_bfloat162*>(beta)  : nullptr;
        __nv_bfloat162* yv = reinterpret_cast<__nv_bfloat162*>(yr);
        int hcols = cols >> 1;
        for (int i = tid; i < hcols; i += bdim) {
            float2 v = bfloat162_to_float2(xv[i]);
            float2 g = gv ? bfloat162_to_float2(gv[i]) : make_float2(1.0f, 1.0f);
            float2 b = bv ? bfloat162_to_float2(bv[i]) : make_float2(0.0f, 0.0f);
            float o0 = ((v.x - mu) * rstd) * g.x + b.x;
            float o1 = ((v.y - mu) * rstd) * g.y + b.y;
            yv[i] = float2_to_bfloat162(o0, o1);
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float g = gamma ? __bfloat162float(gamma[i]) : 1.0f;
            float b = beta  ? __bfloat162float(beta[i])  : 0.0f;
            yr[i] = __float2bfloat16(
                ((__bfloat162float(xr[i]) - mu) * rstd) * g + b);
        }
    }
}

// =============================================================================
// bf16 backward kernel: input gradients
// =============================================================================
__global__ void vln_bwd_input_bf16_kernel(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ x,
    const float*          __restrict__ mean,
    const float*          __restrict__ rstd,
    const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ dx,
    int cols, bool vectorized)
{
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    const __nv_bfloat16* dy_row = dy + (int64_t)row * cols;
    const __nv_bfloat16* x_row  = x  + (int64_t)row * cols;
    __nv_bfloat16*       dx_row = dx + (int64_t)row * cols;
    float mu = mean[row], rs = rstd[row];

    float sum1 = 0.0f, sum2 = 0.0f;
    if (vectorized) {
        const __nv_bfloat162* dv = reinterpret_cast<const __nv_bfloat162*>(dy_row);
        const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(x_row);
        const __nv_bfloat162* gv = gamma ? reinterpret_cast<const __nv_bfloat162*>(gamma) : nullptr;
        int hcols = cols >> 1;
        for (int i = tid; i < hcols; i += bdim) {
            float2 d = bfloat162_to_float2(dv[i]);
            float2 xx = bfloat162_to_float2(xv[i]);
            float2 g = gv ? bfloat162_to_float2(gv[i]) : make_float2(1.0f, 1.0f);
            float n0 = (xx.x - mu) * rs, n1 = (xx.y - mu) * rs;
            sum1 += d.x*g.x + d.y*g.y;
            sum2 += d.x*g.x*n0 + d.y*g.y*n1;
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float g = gamma ? __bfloat162float(gamma[i]) : 1.0f;
            float d = __bfloat162float(dy_row[i]);
            float n = (__bfloat162float(x_row[i]) - mu) * rs;
            sum1 += d*g; sum2 += d*g*n;
        }
    }

    float total1 = vln_block_reduce(sum1, smem);
    if (tid == 0) smem[32] = total1;
    __syncthreads();
    float total2 = vln_block_reduce(sum2, smem);
    total1 = smem[32];
    float inv_cols = 1.0f / cols;

    if (vectorized) {
        const __nv_bfloat162* dv = reinterpret_cast<const __nv_bfloat162*>(dy_row);
        const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(x_row);
        const __nv_bfloat162* gv = gamma ? reinterpret_cast<const __nv_bfloat162*>(gamma) : nullptr;
        __nv_bfloat162* dxv = reinterpret_cast<__nv_bfloat162*>(dx_row);
        int hcols = cols >> 1;
        for (int i = tid; i < hcols; i += bdim) {
            float2 d = bfloat162_to_float2(dv[i]);
            float2 xx = bfloat162_to_float2(xv[i]);
            float2 g = gv ? bfloat162_to_float2(gv[i]) : make_float2(1.0f, 1.0f);
            float n0 = (xx.x - mu) * rs, n1 = (xx.y - mu) * rs;
            float o0 = rs * (d.x*g.x - (total1 + n0*total2) * inv_cols);
            float o1 = rs * (d.y*g.y - (total1 + n1*total2) * inv_cols);
            dxv[i] = float2_to_bfloat162(o0, o1);
        }
    } else {
        for (int i = tid; i < cols; i += bdim) {
            float g  = gamma ? __bfloat162float(gamma[i]) : 1.0f;
            float d  = __bfloat162float(dy_row[i]);
            float nx = (__bfloat162float(x_row[i]) - mu) * rs;
            dx_row[i] = __float2bfloat16(rs * (d*g - (total1 + nx*total2) * inv_cols));
        }
    }
}

// =============================================================================
// bf16 backward kernel: gamma/beta gradients (float32 output)
// =============================================================================
__global__ void vln_bwd_gamma_beta_bf16_kernel(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ x,
    const float*          __restrict__ mean,
    const float*          __restrict__ rstd,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows, int cols)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float s_dg[8][32];
    __shared__ float s_db[8][32];

    for (int col_base = blockIdx.x * 32; col_base < cols; col_base += gridDim.x * 32) {
        int col = col_base + tx;
        float dg_acc = 0.0f, db_acc = 0.0f;
        if (col < cols) {
            for (int row = blockIdx.y * 8 + ty; row < rows; row += gridDim.y * 8) {
                float gy   = __bfloat162float(dy[row * cols + col]);
                float xval = __bfloat162float(x[row * cols + col]);
                float nx   = (xval - mean[row]) * rstd[row];
                db_acc += gy; dg_acc += gy * nx;
            }
        }
        s_dg[ty][tx] = dg_acc; s_db[ty][tx] = db_acc;
        __syncthreads();
        if (ty == 0 && col < cols) {
            float fg = 0.0f, fb = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) { fg += s_dg[i][tx]; fb += s_db[i][tx]; }
            if (dgamma) atomicAdd(&dgamma[col], fg);
            if (dbeta)  atomicAdd(&dbeta[col],  fb);
        }
        __syncthreads();
    }
}

// =============================================================================
// Launchers: bf16
// =============================================================================

void vln_forward_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
    __nv_bfloat16* y, float* mean_out, float* rstd_out, int rows, int cols, float eps)
{
    bool vec = (cols % 2 == 0);
    int threads = std::min(std::max(cols / (vec ? 2 : 1), 32), 256);
    threads = ((threads + 31) / 32) * 32;
    size_t smem = 32 * sizeof(float);
    vln_fwd_bf16_kernel<<<rows, threads, smem>>>(
        x, gamma, beta, y, mean_out, rstd_out, cols, eps, vec);
}

void vln_backward_bf16(
    const __nv_bfloat16* dy, const __nv_bfloat16* x,
    const float* mean, const float* rstd, const __nv_bfloat16* gamma,
    __nv_bfloat16* dx, float* dgamma, float* dbeta, int rows, int cols)
{
    bool vec = (cols % 2 == 0);

    if (dgamma || dbeta) {
        if (dgamma) cudaMemset(dgamma, 0, cols * sizeof(float));
        if (dbeta)  cudaMemset(dbeta,  0, cols * sizeof(float));
        dim3 threads(32, 8);
        int bx = (cols + 31) / 32;
        int by = std::max(1, std::min(32, 128 / bx));
        vln_bwd_gamma_beta_bf16_kernel<<<dim3(bx, by), threads>>>(
            dy, x, mean, rstd, dgamma, dbeta, rows, cols);
    }

    if (dx) {
        int threads = std::min(std::max(cols / (vec ? 2 : 1), 32), 512);
        threads = ((threads + 31) / 32) * 32;
        size_t smem = 34 * sizeof(float);
        vln_bwd_input_bf16_kernel<<<rows, threads, smem>>>(
            dy, x, mean, rstd, gamma, dx, cols, vec);
    }
}

} // namespace cuda
} // namespace OwnTensor
