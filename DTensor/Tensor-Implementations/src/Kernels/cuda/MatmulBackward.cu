#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <string>
#include <mutex>

#include "ops/MatmulBackward.cuh"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "utils/Profiler.h"

namespace OwnTensor {

// cuBLAS handle management - thread-safe singleton per device
static std::mutex cublas_mutex_bwd;
static cublasHandle_t g_cublas_handles_bwd[8] = {nullptr};

static cublasHandle_t get_cublas_handle_bwd(int device = 0) {
   if (g_cublas_handles_bwd[device] == nullptr) {
      std::lock_guard<std::mutex> lock(cublas_mutex_bwd);
      if (g_cublas_handles_bwd[device] == nullptr) {
         cudaSetDevice(device);
         cublasCreate(&g_cublas_handles_bwd[device]);
         // Enable TF32 for FP32 matmuls on Ampere+ GPUs for significant speedup
         cublasSetMathMode(g_cublas_handles_bwd[device], CUBLAS_TF32_TENSOR_OP_MATH);
      }
   }
   return g_cublas_handles_bwd[device];
}

using namespace nvcuda;

// ============================================================================
// METADATA & CONSTANTS (Reuse from forward)
// ============================================================================

struct BackwardMetadata {
   int a_shape[8], b_shape[8], grad_out_shape[8];
   int a_strides[8], b_strides[8], grad_out_strides[8];
   int grad_a_strides[8], grad_b_strides[8];
   int a_ndim, b_ndim, grad_out_ndim;
};

__device__ void compute_batch_offset_bwd(int batch_idx, const int* shape, const int* strides, 
                                          int ndim, const int* out_shape, int out_ndim, int& offset) {
   offset = 0; 
   if (out_ndim <= 2) return;
   int temp_batch = batch_idx;
   for (int dim = out_ndim - 3; dim >= 0; --dim) {
      int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
      temp_batch /= b_dim_sz;
      int c_dim = dim - (out_ndim - ndim);
      if (c_dim >= 0 && c_dim < ndim - 2) 
         offset += (int64_t)((shape[c_dim] > 1) ? b_coord : 0) * strides[c_dim];
   }
}

constexpr int BACKWARD_PAD = 8;

// ============================================================================
// FP32 BACKWARD KERNEL: grad_A = grad_output @ B^T
// Computes: C[M,K] = A[M,N] @ B[K,N]^T = A[M,N] @ B^T[N,K]
// Note: We read B in transposed order (columns become rows)
// BM=128, BN=128, BK=16, TM=4, TN=4, Threads=1024
// ============================================================================

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_backward_dA_fp32(
    const float* __restrict__ grad_out,  // [M, N] - grad of output
    const float* __restrict__ B,         // [K, N] - second input of forward, read transposed
    float* __restrict__ grad_A,          // [M, K] - gradient for A
    int M, int N, int K,                 // M=rows of grad_out, N=cols of grad_out (=cols of B), K=rows of B
    int total_batches,
    BackwardMetadata meta) 
{
   const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
   if (b_idx >= total_batches) return;
   const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
   int go_offset = 0, b_offset = 0, ga_offset = 0;
   compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, go_offset);
   compute_batch_offset_bwd(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, b_offset);
   compute_batch_offset_bwd(b_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
   const float *GO = grad_out + go_offset;
   const float *Bp = B + b_offset;
   float *GA = grad_A + ga_offset;
   
   // Strides for grad_output [M, N]
   int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
   int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   
   // Strides for B [K, N] - we read it transposed as B^T [N, K]
   int s_b_k = meta.b_strides[meta.b_ndim-2];
   int s_b_n = meta.b_strides[meta.b_ndim-1];
   
   // Strides for grad_A [M, K]
   int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
   int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

   __shared__ float As[2][BK][BM + BACKWARD_PAD];  // grad_out tile: [BK, BM] transposed storage
   __shared__ float Bs[2][BK][BN + BACKWARD_PAD];  // B^T tile: [BK, BN]
   float results[16] = {0.0f}, regM[4], regN[4];

   // Load tiles with double buffering
   // For grad_A = grad_out @ B^T:
   //   We iterate over N (the shared dimension between grad_out and B^T)
   //   grad_out is [M, N], B^T is [N, K]
   //   So we load BK columns of grad_out into shared As
   //   And BK rows of B^T (= BK columns of B) into shared Bs
   
   auto load_tiles = [&](int no, int idx) {
      // Load grad_out[by*BM : by*BM+BM, no : no+BK] into As (transposed for coalescing)
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BK, c = li % BK;
         int gm = by * BM + r, gn = no + c;
         As[idx][c][r] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
      }
      
      // Load B^T[no : no+BK, bx*BN : bx*BN+BN] = B[bx*BN : bx*BN+BN, no : no+BK]^T
      // B is [K, N], B^T is [N, K]. We want B^T[n, k] = B[k, n]
      // So for k in [bx*BN, bx*BN+BN) and n in [no, no+BK)
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BN, c = li % BN;
         int gn = no + r, gk = bx * BN + c;
         // B^T[gn, gk] = B[gk, gn]
         Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : 0.0f;
      }
   };

   int wi = 0; 
   load_tiles(0, wi); 
   __syncthreads();
   
   for (int bk = 0; bk < N; bk += BK) {
      int ri = wi; 
      wi = 1 - wi;
      if (bk + BK < N) load_tiles(bk + BK, wi);
      
      #pragma unroll
      for (int d = 0; d < BK; d++) {
         #pragma unroll
         for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
         #pragma unroll
         for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
         #pragma unroll
         for (int i = 0; i < 4; i++) 
            for (int j = 0; j < 4; j++) 
               results[i*4+j] += regM[i] * regN[j];
      }
      __syncthreads();
   }
   
   // Store results to grad_A
   for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         int r = by * BM + tRow * 4 + i;
         int c = bx * BN + tCol * 4 + j;
         if (r < M && c < K) {
            GA[r * s_ga_m + c * s_ga_k] = results[i*4+j];
         }
      }
   }
}

// ============================================================================
// FP32 BACKWARD KERNEL: grad_B = A^T @ grad_output
// Computes: C[K,N] = A[M,K]^T @ B[M,N] = A^T[K,M] @ B[M,N]
// Note: We read A in transposed order
// ============================================================================

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_backward_dB_fp32(
    const float* __restrict__ A,         // [M, K] - first input of forward, read transposed
    const float* __restrict__ grad_out,  // [M, N] - grad of output
    float* __restrict__ grad_B,          // [K, N] - gradient for B
    int M, int K, int N,                 // M=rows of A, K=cols of A, N=cols of grad_out
    int total_batches,
    BackwardMetadata meta)
{
   const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
   if (b_idx >= total_batches) return;
   const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
   int a_offset = 0, go_offset = 0, gb_offset = 0;
   compute_batch_offset_bwd(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, a_offset);
   compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, go_offset);
   compute_batch_offset_bwd(b_idx, meta.b_shape, meta.grad_b_strides, meta.b_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, gb_offset);
   
   const float *Ap = A + a_offset;
   const float *GO = grad_out + go_offset;
   float *GB = grad_B + gb_offset;
   
   // Strides for A [M, K] - we read it transposed as A^T [K, M]
   int s_a_m = meta.a_strides[meta.a_ndim-2];
   int s_a_k = meta.a_strides[meta.a_ndim-1];
   
   // Strides for grad_output [M, N]
   int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
   int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   
   // Strides for grad_B [K, N]
   int s_gb_k = meta.grad_b_strides[meta.b_ndim-2];
   int s_gb_n = meta.grad_b_strides[meta.b_ndim-1];

   __shared__ float As[2][BK][BM + BACKWARD_PAD];  // A^T tile: [BK, BM]
   __shared__ float Bs[2][BK][BN + BACKWARD_PAD];  // grad_out tile: [BK, BN]
   float results[16] = {0.0f}, regM[4], regN[4];

   // For grad_B = A^T @ grad_out:
   //   We iterate over M (the shared dimension)
   //   A^T is [K, M], grad_out is [M, N]
   //   So we load BK rows of A^T (= BK columns of A) into shared As
   //   And BK rows of grad_out into shared Bs
   
   auto load_tiles = [&](int mo, int idx) {
      // Load A^T[by*BM : by*BM+BM, mo : mo+BK] = A[mo : mo+BK, by*BM : by*BM+BM]^T
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BK, c = li % BK;
         int gk = by * BM + r, gm = mo + c;
         // A^T[gk, gm] = A[gm, gk]
         As[idx][c][r] = (gk < K && gm < M) ? Ap[gm * s_a_m + gk * s_a_k] : 0.0f;
      }
      
      // Load grad_out[mo : mo+BK, bx*BN : bx*BN+BN] into Bs
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BN, c = li % BN;
         int gm = mo + r, gn = bx * BN + c;
         Bs[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
      }
   };

   int wi = 0; 
   load_tiles(0, wi); 
   __syncthreads();
   
   for (int bk = 0; bk < M; bk += BK) {
      int ri = wi; 
      wi = 1 - wi;
      if (bk + BK < M) load_tiles(bk + BK, wi);
      
      #pragma unroll
      for (int d = 0; d < BK; d++) {
         #pragma unroll
         for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
         #pragma unroll
         for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
         #pragma unroll
         for (int i = 0; i < 4; i++) 
            for (int j = 0; j < 4; j++) 
               results[i*4+j] += regM[i] * regN[j];
      }
      __syncthreads();
   }
   
   // Store results to grad_B
   for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         int r = by * BM + tRow * 4 + i;
         int c = bx * BN + tCol * 4 + j;
         if (r < K && c < N) {
            GB[r * s_gb_k + c * s_gb_n] = results[i*4+j];
         }
      }
   }
}

// ============================================================================
// FP16 BACKWARD KERNEL: grad_A = grad_output @ B^T (WMMA)
// ============================================================================

template<int BM, int BN, int BK, int WM, int WN>
__global__ void matmul_backward_dA_fp16(
    const __half* __restrict__ grad_out,
    const __half* __restrict__ B,
    __half* __restrict__ grad_A,
    int M, int N, int K,
    int total_batches,
    BackwardMetadata meta)
{
   const int batch_idx = blockIdx.z; 
   if (batch_idx >= total_batches) return;
   const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4, warp_col = warp_id % 4;
   
   int go_offset = 0, b_offset = 0, ga_offset = 0;
   compute_batch_offset_bwd(batch_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim,
                            meta.grad_out_shape, meta.grad_out_ndim, go_offset);
   compute_batch_offset_bwd(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
                            meta.grad_out_shape, meta.grad_out_ndim, b_offset);
   compute_batch_offset_bwd(batch_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim,
                            meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
   const __half *GO = grad_out + go_offset;
   const __half *Bp = B + b_offset;
   __half *GA = grad_A + ga_offset;
   
   int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
   int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   int s_b_k = meta.b_strides[meta.b_ndim-2];
   int s_b_n = meta.b_strides[meta.b_ndim-1];
   int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
   int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

   __shared__ __half As[2][BM][BK + BACKWARD_PAD];
   __shared__ __half Bs[2][BK][BN + BACKWARD_PAD];
   
   wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
   wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
   
   #pragma unroll
   for(int i = 0; i < 2; i++) 
      for(int j = 0; j < 2; j++) 
         wmma::fill_fragment(acc[i][j], __float2half(0.0f));

   auto load_tiles = [&](int no, int idx) {
      for (int i = tid; i < BM * BK; i += 512) {
         int r = i / BK, c = i % BK;
         int gm = blockIdx.y * BM + r, gn = no + c;
         As[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : __float2half(0.0f);
      }
      for (int i = tid; i < BK * BN; i += 512) {
         int r = i / BN, c = i % BN;
         int gn = no + r, gk = blockIdx.x * BN + c;
         Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : __float2half(0.0f);
      }
   };
   
   int wi = 0; 
   load_tiles(0, wi); 
   __syncthreads();
   
   for (int k = 0; k < N; k += BK) {
      int ri = wi; 
      wi = 1 - wi;
      if (k + BK < N) load_tiles(k + BK, wi);
      
      #pragma unroll
      for (int ks = 0; ks < BK; ks += 16) {
         #pragma unroll
         for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks], BK + BACKWARD_PAD);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
               wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16], BN + BACKWARD_PAD);
               wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
            }
         }
      }
      __syncthreads();
   }
   
   __half* sm = reinterpret_cast<__half*>(As);
   #pragma unroll
   for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
         int cr = blockIdx.y * BM + warp_row * WM + i * 16;
         int cc = blockIdx.x * BN + warp_col * WN + j * 16;
         if (cr < M && cc < K) {
            __half* wsm = sm + warp_id * 256;
            wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
            for (int r = 0; r < 16; r++) 
               for (int c = 0; c < 16; c++)
                  if (cr + r < M && cc + c < K) 
                     GA[(cr + r) * s_ga_m + (cc + c) * s_ga_k] = wsm[r * 16 + c];
         }
      }
   }
}

// ============================================================================
// DISPATCH LAYER
// ============================================================================

template<typename T>
void launch_backward_matmul(
    const Tensor& grad_output,
    const Tensor& A,
    const Tensor& B,
    Tensor& grad_A,
    Tensor& grad_B,
    cudaStream_t stream)
{
   const auto& go_sh = grad_output.shape().dims;
   const auto& a_sh = A.shape().dims;
   const auto& b_sh = B.shape().dims;
   
   int go_ndim = go_sh.size();
   int a_ndim = a_sh.size();
   int b_ndim = b_sh.size();
   
   int M = go_sh[go_ndim - 2];  // rows of grad_output
   int N = go_sh[go_ndim - 1];  // cols of grad_output (= cols of B)
   int K_a = a_sh[a_ndim - 1];  // cols of A (= rows of grad_A's cols dim)
   int K_b = b_sh[b_ndim - 2];  // rows of B
   
   int tb = 1;
   for (int i = 0; i < go_ndim - 2; i++) tb *= go_sh[i];
   
   BackwardMetadata meta;
   meta.a_ndim = a_ndim;
   meta.b_ndim = b_ndim;
   meta.grad_out_ndim = go_ndim;
   
   for (int i = 0; i < a_ndim; i++) {
      meta.a_shape[i] = a_sh[i];
      meta.a_strides[i] = A.stride().strides[i];
      meta.grad_a_strides[i] = grad_A.stride().strides[i];
   }
   for (int i = 0; i < b_ndim; i++) {
      meta.b_shape[i] = b_sh[i];
      meta.b_strides[i] = B.stride().strides[i];
      meta.grad_b_strides[i] = grad_B.stride().strides[i];
   }
   for (int i = 0; i < go_ndim; i++) {
      meta.grad_out_shape[i] = go_sh[i];
      meta.grad_out_strides[i] = grad_output.stride().strides[i];
   }
   
   const T* go_ptr = grad_output.data<T>();
   const T* a_ptr = A.data<T>();
   const T* b_ptr = B.data<T>();
   T* ga_ptr = grad_A.data<T>();
   T* gb_ptr = grad_B.data<T>();
   
   // Check stride attributes
   bool go_contiguous = (meta.grad_out_strides[go_ndim-1] == 1);
   bool a_contiguous = (meta.a_strides[a_ndim-1] == 1);
   bool a_transposed = (a_ndim >= 2 && meta.a_strides[a_ndim-2] == 1);
   bool b_contiguous = (meta.b_strides[b_ndim-1] == 1);
   bool b_transposed = (b_ndim >= 2 && meta.b_strides[b_ndim-2] == 1);
   bool ga_contiguous = (meta.grad_a_strides[a_ndim-1] == 1);
   bool gb_contiguous = (meta.grad_b_strides[b_ndim-1] == 1);

   bool supported_for_fast_path = go_contiguous && (a_contiguous || a_transposed) && (b_contiguous || b_transposed) && ga_contiguous && gb_contiguous;

    AUTO_PROFILE_CUDA("Backward::Matmul_CUDA");
    bool grad_a_done = false;
    if constexpr (std::is_same<T, float>::value) {
        if (supported_for_fast_path) {
          int current_device; cudaGetDevice(&current_device);
          cublasHandle_t handle = get_cublas_handle_bwd(current_device);
          cublasSetStream(handle, stream);
          float alpha = 1.0f, beta = 0.0f;
          
          cublasOperation_t opB = b_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
          int ldb = b_contiguous ? meta.b_strides[b_ndim-2] : meta.b_strides[b_ndim-1];
          int ldgo = meta.grad_out_strides[go_ndim-2];
          int ldga = meta.grad_a_strides[a_ndim-2];
          
          long long stride_go = (go_ndim == 3) ? meta.grad_out_strides[0] : (static_cast<long long>(M)*N); // Approx for 4+ dims
          long long stride_b = (b_ndim == 3) ? meta.b_strides[0] : 0; // Broadcast if mismatched
          if (b_ndim == go_ndim) stride_b = (b_contiguous ? meta.b_strides[b_ndim-3] * meta.b_strides[b_ndim-2] : 0); // Crude estimation, trust ndim check
          if (b_ndim < go_ndim) stride_b = 0; // Explicit broadcast
          
          long long stride_ga = (go_ndim == 3) ? meta.grad_a_strides[0] : (static_cast<long long>(M)*K_a);

          cublasStatus_t status;
           if (tb == 1) {
              status = cublasGemmEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_ptr, CUDA_R_32F, ldb, go_ptr, CUDA_R_32F, ldgo, &beta, ga_ptr, CUDA_R_32F, ldga, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
           } else {
              // Handle strided batch for grad_A (preserving batch structure)
              status = cublasGemmStridedBatchedEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_ptr, CUDA_R_32F, ldb, stride_b, go_ptr, CUDA_R_32F, ldgo, stride_go, &beta, ga_ptr, CUDA_R_32F, ldga, stride_ga, tb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
           }
           
          if (status == CUBLAS_STATUS_SUCCESS) grad_a_done = true;
       }
   }
   
   if (!grad_a_done) {
        #ifdef AUTOGRAD_PROFILER_ENABLED
        if (M*N*K_a > 1000000) {
            printf("dA Fallback: M=%d N=%d K=%d, Strides: go[%d,%d], b[%d,%d], ga[%d,%d]\n", 
                   M, N, K_a, meta.grad_out_strides[go_ndim-2], meta.grad_out_strides[go_ndim-1],
                   meta.b_strides[b_ndim-2], meta.b_strides[b_ndim-1],
                   meta.grad_a_strides[a_ndim-2], meta.grad_a_strides[a_ndim-1]);
        }
        #endif
       // Fallback
       if constexpr (std::is_same<T, float>::value)
          matmul_backward_dA_fp32<128, 128, 16, 4, 4><<<dim3((K_a+127)/128, (M+127)/128, tb), 1024, 0, stream>>>(go_ptr, b_ptr, ga_ptr, M, N, K_a, tb, meta);
       else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value)
          matmul_backward_dA_fp16<128, 128, 32, 32, 32><<<dim3((K_a+127)/128, (M+127)/128, tb), 512, 0, stream>>>(reinterpret_cast<const __half*>(go_ptr), reinterpret_cast<const __half*>(b_ptr), reinterpret_cast<__half*>(ga_ptr), M, N, K_a, tb, meta);
   }

   // Launch grad_B kernel: grad_B = A^T @ grad_output
   // Handle Reduction if B dim < Output dim (Broadcast in forward)
   
   bool grad_b_actions_needed = true; // Could optimize if not req grad
   bool grad_b_done = false;
   
   // Check reduction need
   bool need_reduction = (b_ndim < go_ndim) || (tb > 1 && b_ndim == go_ndim && meta.b_shape[0] == 1);
   // Standard broadcast check: if tb > 1 but grad_B is not batched.
   // Assuming grad_B matches B shape.
   if (tb > 1 && b_ndim < go_ndim) need_reduction = true;

   if constexpr (std::is_same<T, float>::value) {
        if (supported_for_fast_path) {
          int current_device; cudaGetDevice(&current_device);
          cublasHandle_t handle = get_cublas_handle_bwd(current_device);
          cublasSetStream(handle, stream);
          float alpha = 1.0f, beta = 0.0f;
          
          cublasOperation_t opA = a_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N; // A^T
          int lda = a_contiguous ? meta.a_strides[a_ndim-2] : meta.a_strides[a_ndim-1]; // Correct stride 
          
          int ldgo = meta.grad_out_strides[go_ndim-2];
          int ldgb = meta.grad_b_strides[b_ndim-2];
          
          cublasStatus_t status;
          
          if (need_reduction) {
              // Flatten Batch strategy: Treat A as [TotalRows, K], GO as [TotalRows, N]
              // grad_B = A^T @ GO.
              // A^T [K, TotalRows]. GO [TotalRows, N]. Result [K, N].
              int M_total = M * tb;
              
              status = cublasGemmEx(handle, CUBLAS_OP_N, opA, N, K_a, M_total, &alpha, go_ptr, CUDA_R_32F, ldgo, a_ptr, CUDA_R_32F, lda, &beta, gb_ptr, CUDA_R_32F, ldgb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
              
          } else {
             // Batched Strategy
             long long stride_a = (a_ndim == 3) ? meta.a_strides[0] : (static_cast<long long>(M)*K_a); 
             long long stride_go = (go_ndim == 3) ? meta.grad_out_strides[0] : (static_cast<long long>(M)*N);
             long long stride_gb = (b_ndim == 3) ? meta.grad_b_strides[0] : (static_cast<long long>(K_a)*N);
             
             status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA, N, K_a, M, &alpha, go_ptr, CUDA_R_32F, ldgo, stride_go, a_ptr, CUDA_R_32F, lda, stride_a, &beta, gb_ptr, CUDA_R_32F, ldgb, stride_gb, tb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
          }
          if (status == CUBLAS_STATUS_SUCCESS) grad_b_done = true;
       }
   }
   
   if (!grad_b_done) {
        #ifdef AUTOGRAD_PROFILER_ENABLED
        if (M*N*K_a > 1000000) {
            printf("dB Fallback: M=%d K=%d N=%d, Strides: a[%d,%d], go[%d,%d], gb[%d,%d]\n", 
                   M, K_a, N, meta.a_strides[a_ndim-2], meta.a_strides[a_ndim-1],
                   meta.grad_out_strides[go_ndim-2], meta.grad_out_strides[go_ndim-1],
                   meta.grad_b_strides[b_ndim-2], meta.grad_b_strides[b_ndim-1]);
        }
        #endif
       // Fallback
       if constexpr (std::is_same<T, float>::value)
          matmul_backward_dB_fp32<128, 128, 16, 4, 4><<<dim3((N+127)/128, (K_a+127)/128, tb), 1024, 0, stream>>>(a_ptr, go_ptr, gb_ptr, M, K_a, N, tb, meta);
       // FP16 fallback not implemented for dB yet
   }
   
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
      throw std::runtime_error("Backward kernel failed: " + std::string(cudaGetErrorString(err)));
   }
}

void cuda_matmul_backward(
    const Tensor& grad_output,
    const Tensor& A,
    const Tensor& B,
    Tensor& grad_A,
    Tensor& grad_B,
    cudaStream_t stream)
{
   dispatch_by_dtype(A.dtype(), [&](auto d) {
      using T = decltype(d);
      if constexpr (std::is_same<T, float>::value) {
         launch_backward_matmul<T>(grad_output, A, B, grad_A, grad_B, stream);
      } else {
         throw std::runtime_error("cuda_matmul_backward: Unsupported type (currently only float supported)");
      }
   });
}

} // namespace OwnTensor

#endif
