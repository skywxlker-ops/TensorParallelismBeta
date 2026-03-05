

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "utils/Profiler.h"
#include <mma.h>
#include <cublas_v2.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <string>
#include <mutex>

#include "ops/Matmul.cuh"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "ops/Kernels.h"

namespace OwnTensor {

// cuBLAS handle management - thread-safe singleton per device
static std::mutex cublas_mutex;
static cublasHandle_t g_cublas_handles[8] = {nullptr};

static cublasHandle_t get_cublas_handle(int device = 0) {
   if (g_cublas_handles[device] == nullptr) {
      std::lock_guard<std::mutex> lock(cublas_mutex);
      if (g_cublas_handles[device] == nullptr) {
         cudaSetDevice(device);
         cublasCreate(&g_cublas_handles[device]);
         // Enable TF32 for FP32 matmuls on Ampere+ GPUs for significant speedup
         cublasSetMathMode(g_cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH);
      }
   }
   return g_cublas_handles[device];
}

using namespace nvcuda;

// ============================================================================
// METADATA & CONSTANTS
// ============================================================================

struct MatmulMetadata {
   int a_shape[8], b_shape[8], out_shape[8];
   int a_strides[8], b_strides[8], out_strides[8];
   int a_ndim, b_ndim, out_ndim;
};

__device__ void compute_batch_offset(int batch_idx, const int* shape, const int* strides, int ndim, const int* out_shape, int out_ndim, int& offset) {
   offset = 0; if (out_ndim <= 2) return;
   int temp_batch = batch_idx;
   for (int dim = out_ndim - 3; dim >= 0; --dim) {
      int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
      temp_batch /= b_dim_sz;
      int c_dim = dim - (out_ndim - ndim);
      if (c_dim >= 0 && c_dim < ndim - 2) offset += (int64_t)((shape[c_dim] > 1) ? b_coord : 0) * strides[c_dim];
   }
}

constexpr int PAD = 8;

// ============================================================================
// ADDMM HELPERS
// ============================================================================

template<typename T>
__global__ void broadcast_scale_kernel(T* output, const T* input, float scale, int64_t total_elements, int64_t input_size, int64_t rows, int64_t cols) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16> || std::is_integral_v<T>) {
            if (input_size == 1) { // Scalar
                output[idx] = (T)((float)input[0] * scale);
            } else if (input_size == cols) { // Row broadcast (bias-like)
                output[idx] = (T)((float)input[idx % cols] * scale);
            } else if (input_size == total_elements) { // Same shape
                output[idx] = (T)((float)input[idx] * scale);
            } else {
                output[idx] = (T)0.0f;
            }
        } else {
            // For complex types, we'd need a proper implementation, but for now we just skip or do basic copy
            if (input_size == total_elements) {
                output[idx] = input[idx];
            } else {
                output[idx] = (T)0.0f;
            }
        }
    }
}

template<typename T>
void launch_broadcast_scale(T* output, const T* input, float scale, int64_t rows, int64_t cols, int64_t input_size, cudaStream_t stream) {
    int64_t total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    broadcast_scale_kernel<T><<<blocks, threads, 0, stream>>>(output, input, scale, total, input_size, rows, cols);
}

template<typename T>
__global__ void add_scaled_kernel_typed(T* out, const T* m, float a, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            out[i] += (T)(a * (float)m[i]);
        } else {
            // Fallback for types that might not support float multiplication directly or need different logic
            // For complex types, we'd need a proper implementation, but for now we just skip or do basic add
            out[i] += (T)m[i]; 
        }
    }
}

// ============================================================================
// FP16 WMMA KERNEL (17 TFLOPS Version)
// BM=128, BN=128, BK=32, Threads=512
// ============================================================================

template<int BM, int BN, int BK, int WM, int WN>
__global__ void matmul_fp16_optimized(const __half* __restrict__ A, const __half* __restrict__ B, __half* __restrict__ C, int M, int N, int K, int total_batches, MatmulMetadata meta) {
   const int batch_idx = blockIdx.z; if (batch_idx >= total_batches) return;
   const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4, warp_col = warp_id % 4;
   int ao, bo, co;
   compute_batch_offset(batch_idx, meta.a_shape, meta.a_strides, meta.a_ndim, meta.out_shape, meta.out_ndim, ao);
   compute_batch_offset(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim, meta.out_shape, meta.out_ndim, bo);
   compute_batch_offset(batch_idx, meta.out_shape, meta.out_strides, meta.out_ndim, meta.out_shape, meta.out_ndim, co);
   const __half *Ap = A + ao, *Bp = B + bo; __half *Cp = C + co;
   int s_am = meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1];
   int s_bk = meta.b_strides[meta.b_ndim-2], s_bn = meta.b_strides[meta.b_ndim-1];
   int s_cm = meta.out_strides[meta.out_ndim-2], s_cn = meta.out_strides[meta.out_ndim-1];

   __shared__ __half As[2][BM][BK + PAD], Bs[2][BK][BN + PAD];
   wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
   wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
   #pragma unroll
   for(int i=0; i<2; i++) for(int j=0; j<2; j++) wmma::fill_fragment(acc[i][j], __float2half(0.0f));

   auto load_tiles = [&](int ko, int idx) {
      for (int i = tid; i < BM * BK; i += 512) {
         int r = i / BK, c = i % BK;
         As[idx][r][c] = (blockIdx.y*BM+r < M && ko+c < K) ? Ap[(blockIdx.y*BM+r)*s_am + (ko+c)*s_ak] : __float2half(0.0f);
      }
      for (int i = tid; i < BK * BN; i += 512) {
         int r = i / BN, c = i % BN;
         Bs[idx][r][c] = (ko+r < K && blockIdx.x*BN+c < N) ? Bp[(ko+r)*s_bk + (blockIdx.x*BN+c)*s_bn] : __float2half(0.0f);
      }
   };
   int wi = 0; load_tiles(0, wi); __syncthreads();
   for (int k = 0; k < K; k += BK) {
      int ri = wi; wi = 1 - wi;
      if (k + BK < K) load_tiles(k + BK, wi);
      #pragma unroll
      for (int ks = 0; ks < BK; ks += 16) {
         #pragma unroll
         for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks], BK+PAD);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
               wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16], BN+PAD);
               wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
            }
         }
      }
      __syncthreads();
   }
   __half* sm = reinterpret_cast<__half*>(As);
   #pragma unroll
   for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
      int cr = blockIdx.y*BM + warp_row*WM + i*16, cc = blockIdx.x*BN + warp_col*WN + j*16;
      if (cr < M && cc < N) {
         __half* wsm = sm + warp_id * 256;
         wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
         for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++)
            if (cr+r < M && cc+c < N) Cp[(cr+r)*s_cm + (cc+c)*s_cn] = wsm[r*16+c];
      }
   }
}

// ============================================================================
// BF16 WMMA KERNEL
// ============================================================================

template<int BM, int BN, int BK, int WM, int WN>
__global__ void matmul_bf16_optimized(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, int M, int N, int K, int total_batches, MatmulMetadata meta) {
   const int batch_idx = blockIdx.z; if (batch_idx >= total_batches) return;
   const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4, warp_col = warp_id % 4;
   int ao, bo, co;
   compute_batch_offset(batch_idx, meta.a_shape, meta.a_strides, meta.a_ndim, meta.out_shape, meta.out_ndim, ao);
   compute_batch_offset(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim, meta.out_shape, meta.out_ndim, bo);
   compute_batch_offset(batch_idx, meta.out_shape, meta.out_strides, meta.out_ndim, meta.out_shape, meta.out_ndim, co);
   const __nv_bfloat16 *Ap = A + ao, *Bp = B + bo; __nv_bfloat16 *Cp = C + co;
   int s_am = meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1];
   int s_bk = meta.b_strides[meta.b_ndim-2], s_bn = meta.b_strides[meta.b_ndim-1];
   int s_cm = meta.out_strides[meta.out_ndim-2], s_cn = meta.out_strides[meta.out_ndim-1];

   __shared__ __nv_bfloat16 As[2][BM][BK + PAD], Bs[2][BK][BN + PAD];
   wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
   #pragma unroll
   for(int i=0; i<2; i++) for(int j=0; j<2; j++) wmma::fill_fragment(acc[i][j], 0.0f);

   auto load_tiles = [&](int ko, int idx) {
      for (int i = tid; i < BM * BK; i += 512) {
         int r = i / BK, c = i % BK;
         As[idx][r][c] = (blockIdx.y*BM+r < M && ko+c < K) ? Ap[(blockIdx.y*BM+r)*s_am + (ko+c)*s_ak] : __float2bfloat16(0.0f);
      }
      for (int i = tid; i < BK * BN; i += 512) {
         int r = i / BN, c = i % BN;
         Bs[idx][r][c] = (ko+r < K && blockIdx.x*BN+c < N) ? Bp[(ko+r)*s_bk + (blockIdx.x*BN+c)*s_bn] : __float2bfloat16(0.0f);
      }
   };
   int wi = 0; load_tiles(0, wi); __syncthreads();
   for (int k = 0; k < K; k += BK) {
      int ri = wi; wi = 1 - wi;
      if (k + BK < K) load_tiles(k + BK, wi);
      #pragma unroll
      for (int ks = 0; ks < BK; ks += 16) {
         #pragma unroll
         for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks], BK+PAD);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
               wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16], BN+PAD);
               wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
            }
         }
      }
      __syncthreads();
   }
   float* sm = reinterpret_cast<float*>(As);
   #pragma unroll
   for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
      int cr = blockIdx.y*BM + warp_row*WM + i*16, cc = blockIdx.x*BN + warp_col*WN + j*16;
      if (cr < M && cc < N) {
         float* wsm = sm + warp_id * 256;
         wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
         for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++)
            if (cr+r < M && cc+c < N) Cp[(cr+r)*s_cm + (cc+c)*s_cn] = __float2bfloat16(wsm[r*16+c]);
      }
   }
}

// ============================================================================
// FP32 KERNEL (5 TFLOPS Version)
// BM=128, BN=128, BK=16, TM=4, TN=4, Threads=1024
// ============================================================================

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_fp32_optimized(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K, int total_batches, MatmulMetadata meta) {
   const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
   if (b_idx >= total_batches) return;
   const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   int ao, bo, co;
   compute_batch_offset(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim, meta.out_shape, meta.out_ndim, ao);
   compute_batch_offset(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim, meta.out_shape, meta.out_ndim, bo);
   compute_batch_offset(b_idx, meta.out_shape, meta.out_strides, meta.out_ndim, meta.out_shape, meta.out_ndim, co);
   const float *Ap = A + ao, *Bp = B + bo; float *Cp = C + co;
   int s_am = meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1], s_bk = meta.b_strides[meta.b_ndim-2], s_bn = meta.b_strides[meta.b_ndim-1], s_cm = meta.out_strides[meta.out_ndim-2], s_cn = meta.out_strides[meta.out_ndim-1];

   __shared__ float As[2][BK][BM + PAD];
   __shared__ float Bs[2][BK][BN + PAD];
   float results[16] = {0.0f}, regM[4], regN[4];

   auto load_tiles = [&](int ko, int idx) {
      if (s_ak == 1 && s_bn == 1) { // Row-Major Contiguous
         #pragma unroll
         for (int i = 0; i < 2; i++) {
            int li = tid + i * 1024, r = li / 16, c = li % 16;
            As[idx][c][r] = (by*128+r < M && ko+c < K) ? Ap[(by*128+r)*s_am + (ko+c)] : 0.0f;
         }
         #pragma unroll
         for (int i = 0; i < 2; i++) {
            int li = tid + i * 1024, r = li / 128, c = li % 128;
            Bs[idx][r][c] = (ko+r < K && bx*128+c < N) ? Bp[(ko+r)*s_bk + (bx*128+c)] : 0.0f;
         }
      } else { // Generic
         #pragma unroll
         for (int i = 0; i < 2; i++) {
            int li = tid + i * 1024, r = li / BK, c = li % BK;
            As[idx][c][r] = (by*BM+r < M && ko+c < K) ? Ap[(by*BM+r)*s_am + (ko+c)*s_ak] : 0.0f;
         }
         #pragma unroll
         for (int i = 0; i < 2; i++) {
            int li = tid + i * 1024, r = li / BN, c = li % BN;
            Bs[idx][r][c] = (ko+r < K && bx*BN+c < N) ? Bp[(ko+r)*s_bk + (bx*BN+c)*s_bn] : 0.0f;
         }
      }
   };

   int wi = 0; load_tiles(0, wi); __syncthreads();
   for (int bk = 0; bk < K; bk += BK) {
      int ri = wi; wi = 1 - wi;
      if (bk + BK < K) load_tiles(bk + BK, wi);
      #pragma unroll
      for (int d = 0; d < BK; d++) {
         #pragma unroll
         for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
         #pragma unroll
         for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
         #pragma unroll
         for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) results[i*4+j] += regM[i] * regN[j];
      }
      __syncthreads();
   }
   for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
      int r = by*BM + tRow*4 + i, c = bx*BN + tCol*4 + j;
      if (r < M && c < N) Cp[r*s_cm + c*s_cn] = results[i*4+j];
   }
}

// ============================================================================
// FP64 KERNEL (64x64x8 Tile, 256 Threads, 4x4 Tiling)
// ============================================================================

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_fp64_optimized(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int M, int N, int K, int total_batches, MatmulMetadata meta) {
   const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
   if (b_idx >= total_batches) return;
   const int tid = threadIdx.x, tC = tid % (BN/TN), tR = tid / (BN/TN);
   int ao, bo, co;
   compute_batch_offset(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim, meta.out_shape, meta.out_ndim, ao);
   compute_batch_offset(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim, meta.out_shape, meta.out_ndim, bo);
   compute_batch_offset(b_idx, meta.out_shape, meta.out_strides, meta.out_ndim, meta.out_shape, meta.out_ndim, co);
   const double *Ap = A + ao, *Bp = B + bo; double *Cp = C + co;
   int s_am = meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1], s_bk = meta.b_strides[meta.b_ndim-2], s_bn = meta.b_strides[meta.b_ndim-1], s_cm = meta.out_strides[meta.out_ndim-2], s_cn = meta.out_strides[meta.out_ndim-1];
   __shared__ double As[BM*BK], Bs[BK*BN];
   double res[16] = {0.0}, rM[4], rN[4];
   for (int bk = 0; bk < K; bk += BK) {
      for (int i = tid; i < BM*BK; i += 256) { int r = i/BK, c = i%BK; As[i] = (by*BM+r < M && bk+c < K) ? Ap[(by*BM+r)*s_am + (bk+c)*s_ak] : 0.0; }
      for (int i = tid; i < BK*BN; i += 256) { int r = i/BN, c = i%BN; Bs[i] = (bk+r < K && bx*BN+c < N) ? Bp[(bk+r)*s_bk + (bk+c)*s_bn] : 0.0; }
      __syncthreads();
      for (int d = 0; d < BK; d++) {
         for (int i = 0; i < 4; i++) rM[i] = As[(tR*4+i)*BK+d];
         for (int i = 0; i < 4; i++) rN[i] = Bs[d*BN+tC*4+i];
         for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) res[i*4+j] += rM[i] * rN[j];
      }
      __syncthreads();
   }
   for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { int r = by*BM+tR*4+i, c = bx*BN+tC*4+j; if (r < M && c < N) Cp[r*s_cm + c*s_cn] = res[i*4+j]; }
}

// ============================================================================
// DISPATCH LAYER
// ============================================================================

template<typename T>
void launch_optimized_matmul(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream) {
   AUTO_PROFILE_CUDA("Forward::Matmul_CUDA");
   const auto& ash = A.shape().dims, &bsh = B.shape().dims, &osh = output.shape().dims;
   int an = ash.size(), bn = bsh.size(), on = osh.size(), M = ash[an-2], K = ash[an-1], N = bsh[bn-1], tb = 1;
   for (int i = 0; i < on - 2; i++) tb *= osh[i];
   MatmulMetadata meta; meta.a_ndim = an; meta.b_ndim = bn; meta.out_ndim = on;
   for (int i = 0; i < an; i++) { meta.a_shape[i] = ash[i]; meta.a_strides[i] = A.stride().strides[i]; }
   for (int i = 0; i < bn; i++) { meta.b_shape[i] = bsh[i]; meta.b_strides[i] = B.stride().strides[i]; }
   for (int i = 0; i < on; i++) { meta.out_shape[i] = osh[i]; meta.out_strides[i] = output.stride().strides[i]; }
   const T* ap = A.data<T>(), *bp = B.data<T>(); T* op = output.data<T>();

   if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value) {
      matmul_fp16_optimized<128, 128, 32, 32, 32><<<dim3((N+127)/128, (M+127)/128, tb), 512, 0, stream>>>(reinterpret_cast<const __half*>(ap), reinterpret_cast<const __half*>(bp), reinterpret_cast<__half*>(op), M, N, K, tb, meta);
   } else if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
      matmul_bf16_optimized<128, 128, 32, 32, 32><<<dim3((N+127)/128, (M+127)/128, tb), 512, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(ap), reinterpret_cast<const __nv_bfloat16*>(bp), reinterpret_cast<__nv_bfloat16*>(op), M, N, K, tb, meta);
   } else if constexpr (std::is_same<T, float>::value) {
      // Use cuBLAS for FP32 matmuls - enables TF32 Tensor Core acceleration
      // Check if tensors are contiguous (row-major) OR transposed (col-major equivalent) for cuBLAS fast path
      
      bool a_contiguous = (meta.a_strides[an-1] == 1);
      bool a_transposed = (an >= 2 && meta.a_strides[an-2] == 1);
      
      bool b_contiguous = (meta.b_strides[bn-1] == 1);
      bool b_transposed = (bn >= 2 && meta.b_strides[bn-2] == 1);
      
      bool out_contiguous = (meta.out_strides[on-1] == 1);
      
      // Use cuBLAS for:
      // 1. 2D matrices (tb=1)
      // 2. Strided Batched matrices (tb>1)
      bool is_supported_layout = (a_contiguous || a_transposed) && (b_contiguous || b_transposed) && out_contiguous;
      bool is_strided_batch = false;
      long long stride_a = 0, stride_b = 0, stride_c = 0;
      
      if (tb > 1 && is_supported_layout) {
          // Calculate Strides for Batches
          // For broadcasted dims (e.g. B is 2D but A is 3D), stride is 0.
          
          // Helper to get batch stride check
          // Assumption: dimensions 0..on-3 are the batch dimensions.
          // Standard layout: Flattened batch.
          
          if (on == 3) {
             stride_c = meta.out_strides[0];
             stride_a = (an == 3) ? meta.a_strides[0] : 0; // Broadcast if an < 3
             stride_b = (bn == 3) ? meta.b_strides[0] : 0; // Broadcast if bn < 3
             is_strided_batch = true;
          } else if (on == 4) {
             // Basic support for [B, H, M, N]
             // We flatten first 2 dims.
             // Stride is effective stride of the flattened index?
             // Only if contiguous in batch dims.
             // For now, only support if simple packing matches.
             // Simplification: Check if stride[0] works for total batch.
             // This assumes strict N, C, H, W structure usually.
             
             stride_c = static_cast<long long>(M) * N; // Estimate contiguous
             stride_a = (an == 4) ? static_cast<long long>(M) * K : 0;
             stride_b = (bn == 4) ? static_cast<long long>(K) * N : 0;
             is_strided_batch = true; 
          }
      }

      if ((tb == 1 || is_strided_batch) && is_supported_layout) {
         int current_device; cudaGetDevice(&current_device);
         cublasHandle_t handle = get_cublas_handle(current_device);
         cublasSetStream(handle, stream);
         
         float alpha = 1.0f, beta = 0.0f;
         
         // Setup A (Second in CuBLAS, since CuBLAS is ColMajor)
         cublasOperation_t opA; 
         int lda;
         if (a_contiguous) { opA = CUBLAS_OP_N; lda = meta.a_strides[an-2]; }
         else { opA = CUBLAS_OP_T; lda = meta.a_strides[an-1]; }
         
         // Setup B (First in CuBLAS)
         cublasOperation_t opB;
         int ldb;
         if (b_contiguous) { opB = CUBLAS_OP_N; ldb = meta.b_strides[bn-2]; }
         else { opB = CUBLAS_OP_T; ldb = meta.b_strides[bn-1]; }
         
         int ldc = meta.out_strides[on-2];
         
         // Logic correction:
         // C = A @ B.
         // CuBLAS (ColMajor): C^T = B^T @ A^T.
         // Pass B as "A", A as "B".
         // 
         // If A is contiguous RM [M, K]: Represents A^T [K, M] CM.
         // We need A^T in the equation. So Op = N.
         // If A is transposed RM [M, K] (stride 1, M): Represents A [K, M] CM.
         // We need A^T. So Op = T.
         //
         // If B is contiguous RM [K, N]: Represents B^T [N, K] CM.
         // We need B^T in the equation. So Op = N.
         // If B is transposed RM [K, N]: Represents B [N, K] CM.
         // We need B^T. So Op = T.
         
         cublasStatus_t status;
         
         if (tb == 1) {
             status = cublasGemmEx(
                handle,
                opB, opA, 
                N, M, K,
                &alpha,
                bp, CUDA_R_32F, ldb,
                ap, CUDA_R_32F, lda,
                &beta,
                op, CUDA_R_32F, ldc,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
             );
         } else {
             status = cublasGemmStridedBatchedEx(
                handle,
                opB, opA,
                N, M, K,
                &alpha,
                bp, CUDA_R_32F, ldb, stride_b,
                ap, CUDA_R_32F, lda, stride_a,
                &beta,
                op, CUDA_R_32F, ldc, stride_c,
                tb,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
             );
         }

         if (status != CUBLAS_STATUS_SUCCESS) {
            // Fallback (e.g. if alignment issues, though Ex handles most)
             matmul_fp32_optimized<128, 128, 16, 4, 4><<<dim3((N+127)/128, (M+127)/128, tb), 1024, 0, stream>>>(ap, bp, op, M, N, K, tb, meta);
         }
      } else {
         // Fallback to custom kernel
         matmul_fp32_optimized<128, 128, 16, 4, 4><<<dim3((N+127)/128, (M+127)/128, tb), 1024, 0, stream>>>(ap, bp, op, M, N, K, tb, meta);
      }
   } else if constexpr (std::is_same<T, double>::value) {
      matmul_fp64_optimized<64, 64, 8, 4, 4><<<dim3((N+63)/64, (M+63)/64, tb), 256, 0, stream>>>(reinterpret_cast<const double*>(ap), reinterpret_cast<const double*>(bp), reinterpret_cast<double*>(op), M, N, K, tb, meta);
   }

   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) throw std::runtime_error("Kernel failed: " + std::string(cudaGetErrorString(err)));
}

void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream) {
   dispatch_by_dtype(A.dtype(), [&](auto d) {
      using T = decltype(d);
      if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, bfloat16_t>::value || std::is_same<T, float>::value || std::is_same<T, double>::value) launch_optimized_matmul<T>(A, B, output, stream);
      else throw std::runtime_error("Unsupported type");
   });
}

void cuda_addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2, float alpha, float beta, Tensor& output, cudaStream_t stream) {
    AUTO_PROFILE_CUDA("Forward::Addmm_CUDA");
    const auto& m1sh = mat1.shape().dims, &m2sh = mat2.shape().dims, &osh = output.shape().dims;
    int m1n = m1sh.size(), m2n = m2sh.size(), on = osh.size();
    int M = m1sh[m1n-2], K = m1sh[m1n-1], N = m2sh[m2n-1];

    // Compute total batch count from output shape (all dims except last 2)
    int tb = 1;
    for (int i = 0; i < on - 2; i++) tb *= osh[i];
    int64_t total_output_elements = output.numel(); // tb * M * N

    // 1. Prepare output buffer: output = beta * input (with broadcasting)
    //    Scale ALL elements of the output, not just M*N
    if (beta != 0.0f) {
        dispatch_by_dtype(output.dtype(), [&](auto d) {
            using T = decltype(d);
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16> || std::is_integral_v<T>) {
                // Use total_output_elements as both rows*cols and total
                // M_total and N stay the same for broadcasting logic (bias is broadcast per [M,N] slice)
                int64_t rows_total = (int64_t)tb * M;
                launch_broadcast_scale<T>(output.data<T>(), input.data<T>(), beta, rows_total, N, input.numel(), stream);
            }
        });
    } else {
        // If beta is 0, input is ignored, zero the entire output
        dispatch_by_dtype(output.dtype(), [&](auto d) {
            using T = decltype(d);
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16> || std::is_integral_v<T>) {
                int threads = 256;
                int blocks = (total_output_elements + threads - 1) / threads;
                // Simple zero kernel
                broadcast_scale_kernel<T><<<blocks, threads, 0, stream>>>(
                    output.data<T>(), output.data<T>(), 0.0f, total_output_elements, total_output_elements, (int64_t)tb * M, N);
            }
        });
    }

    // 2. Add alpha * (mat1 @ mat2) using cuBLAS
    if (output.dtype() == Dtype::Float32) {
        MatmulMetadata meta; meta.a_ndim = m1n; meta.b_ndim = m2n; meta.out_ndim = on;
        for (int i = 0; i < m1n; i++) { meta.a_shape[i] = m1sh[i]; meta.a_strides[i] = mat1.stride().strides[i]; }
        for (int i = 0; i < m2n; i++) { meta.b_shape[i] = m2sh[i]; meta.b_strides[i] = mat2.stride().strides[i]; }
        for (int i = 0; i < on; i++) { meta.out_shape[i] = osh[i]; meta.out_strides[i] = output.stride().strides[i]; }

        bool a_contiguous = (meta.a_strides[m1n-1] == 1);
        bool a_transposed = (m1n >= 2 && meta.a_strides[m1n-2] == 1);
        bool b_contiguous = (meta.b_strides[m2n-1] == 1);
        bool b_transposed = (m2n >= 2 && meta.b_strides[m2n-2] == 1);
        bool out_contiguous = (meta.out_strides[on-1] == 1);

        bool is_supported_layout = (a_contiguous || a_transposed) && (b_contiguous || b_transposed) && out_contiguous;

        // Compute batch strides (mirroring launch_optimized_matmul)
        bool is_strided_batch = false;
        long long stride_a = 0, stride_b = 0, stride_c = 0;

        if (tb > 1 && is_supported_layout) {
            if (on == 3) {
                stride_c = meta.out_strides[0];
                stride_a = (m1n == 3) ? meta.a_strides[0] : 0;
                stride_b = (m2n == 3) ? meta.b_strides[0] : 0;
                is_strided_batch = true;
            } else if (on == 4) {
                stride_c = static_cast<long long>(M) * N;
                stride_a = (m1n == 4) ? static_cast<long long>(M) * K : 0;
                stride_b = (m2n == 4) ? static_cast<long long>(K) * N : 0;
                is_strided_batch = true;
            }
        }

        if ((tb == 1 || is_strided_batch) && is_supported_layout) {
            int current_device; cudaGetDevice(&current_device);
            cublasHandle_t handle = get_cublas_handle(current_device);
            cublasSetStream(handle, stream);

            cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
            int lda = a_contiguous ? meta.a_strides[m1n-2] : meta.a_strides[m1n-1];
            cublasOperation_t opB = b_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
            int ldb = b_contiguous ? meta.b_strides[m2n-2] : meta.b_strides[m2n-1];
            int ldc = meta.out_strides[on-2];

            // output already has beta * input, so we pass cublas_beta = 1.0 to accumulate
            float cublas_beta = 1.0f;

            cublasStatus_t status;

            if (tb == 1) {
                status = cublasGemmEx(
                    handle, opB, opA, N, M, K, &alpha,
                    mat2.data<float>(), CUDA_R_32F, ldb,
                    mat1.data<float>(), CUDA_R_32F, lda,
                    &cublas_beta, output.data<float>(), CUDA_R_32F, ldc,
                    CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );
            } else {
                status = cublasGemmStridedBatchedEx(
                    handle, opB, opA, N, M, K, &alpha,
                    mat2.data<float>(), CUDA_R_32F, ldb, stride_b,
                    mat1.data<float>(), CUDA_R_32F, lda, stride_a,
                    &cublas_beta, output.data<float>(), CUDA_R_32F, ldc, stride_c,
                    tb,
                    CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );
            }

            if (status == CUBLAS_STATUS_SUCCESS) return;
            // Fall through to fallback on failure
        }
    }

    // Fallback: Use existing batched matmul + scaled addition kernel
    Tensor temp_matmul = OwnTensor::matmul(mat1, mat2, stream);

    dispatch_by_dtype(output.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16> || std::is_integral_v<T>) {
            int64_t total = output.numel();
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            add_scaled_kernel_typed<T><<<blocks, threads, 0, stream>>>(output.data<T>(), temp_matmul.data<T>(), alpha, total);
        }
    });
}


} // namespace OwnTensor
// #endif