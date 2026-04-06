#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace OwnTensor {
namespace cuda {

// =============================================================================
// Vectorized LayerNorm — forward and backward for float, fp16, bf16.
//
// Drop-in replacements for layer_norm_forward_cuda / layer_norm_backward_cuda
// from LayerNormKernels.h with the following improvements:
//
//   float path  : float4 vectorized loads/stores, pure warp-shuffle block
//                 reduction (no atomicAdd in forward).
//   fp16  path  : half2 loads (2 elements per instruction), FP32 accumulation.
//   bf16  path  : nv_bfloat162 loads, FP32 accumulation.
//
// Vectorized paths activate when cols % 4 == 0 (float) / cols % 2 == 0
// (fp16/bf16) and pointers are naturally aligned. A scalar fallback (identical
// to the existing kernels) handles all other cases — so correctness is always
// guaranteed regardless of col count.
//
// Signatures mirror LayerNormKernels.h exactly so callers can switch by
// replacing the include + function name.
// =============================================================================

// ---------------------------------------------------------------------------
// float32
// ---------------------------------------------------------------------------

// Forward: x[rows,cols] → y[rows,cols], mean[rows], rstd[rows]
// gamma/beta may be nullptr.
void vln_forward_f32(
    const float* x,
    const float* gamma,
    const float* beta,
    float*       y,
    float*       mean_out,
    float*       rstd_out,
    int rows,
    int cols,
    float eps);

// Backward: dy,x,mean,rstd,gamma → dx, dgamma, dbeta
// dx / dgamma / dbeta may be nullptr.
void vln_backward_f32(
    const float* dy,
    const float* x,
    const float* mean,
    const float* rstd,
    const float* gamma,
    float*       dx,
    float*       dgamma,
    float*       dbeta,
    int rows,
    int cols);

// ---------------------------------------------------------------------------
// fp16  (accumulation in float32)
// ---------------------------------------------------------------------------

void vln_forward_f16(
    const __half* x,
    const __half* gamma,
    const __half* beta,
    __half*       y,
    float*        mean_out,
    float*        rstd_out,
    int rows,
    int cols,
    float eps);

void vln_backward_f16(
    const __half* dy,
    const __half* x,
    const float*  mean,
    const float*  rstd,
    const __half* gamma,
    __half*       dx,
    float*        dgamma,
    float*        dbeta,
    int rows,
    int cols);

// ---------------------------------------------------------------------------
// bf16  (accumulation in float32)
// ---------------------------------------------------------------------------

void vln_forward_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta,
    __nv_bfloat16*       y,
    float*               mean_out,
    float*               rstd_out,
    int rows,
    int cols,
    float eps);

void vln_backward_bf16(
    const __nv_bfloat16* dy,
    const __nv_bfloat16* x,
    const float*         mean,
    const float*         rstd,
    const __nv_bfloat16* gamma,
    __nv_bfloat16*       dx,
    float*               dgamma,
    float*               dbeta,
    int rows,
    int cols);

} // namespace cuda
} // namespace OwnTensor
