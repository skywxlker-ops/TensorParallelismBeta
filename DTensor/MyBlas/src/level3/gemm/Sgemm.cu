#include "mycublas.h"
#include <cuda_runtime.h>
#include <algorithm>

// ============================================================================
// FP32 KERNEL (V9: Final Mastery - float4 Reads + BK Unrolling)
// BM=128, BN=128, BK=16, TM=8, TN=4, Threads=512
// ============================================================================

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 4

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(glob_ptr));
}

__global__ void sgemm_optimized_v9_kernel(
  int M, int N, int K,
  float alpha,
  const float* __restrict__ A, int lda,
  const float* __restrict__ B, int ldb,
  float beta,
  float* __restrict__ C, int ldc)
{
  const int bx = blockIdx.x, by = blockIdx.y;
  const int tid = threadIdx.x;
  const int tCol = tid % 32, tRow = tid / 32;

  const float *dA = A + (by * BM) * lda;
  const float *dB = B + (bx * BN);
  float *dC = C + (by * BM) * ldc + (bx * BN);

  extern __shared__ float s_mem[];
  float* As = s_mem;
  float* Bs = s_mem + 2 * BM * BK;

  float results[TM][TN];
  #pragma unroll
  for(int i=0; i<TM; i++) for(int j=0; j<TN; j++) results[i][j] = 0.0f;

  auto load_tile = [&](int ko, int buf_idx) {
      // V7 Aligned Loading
      int r_a = tid / 4, c_a = (tid % 4) * 4;
      float* sp_a = &As[buf_idx * BM * BK + r_a * BK + c_a];
      if (by * BM + r_a < M && ko + c_a < K) {
          const float* src_a = &dA[r_a * lda + ko + c_a];
          // cp_async_16 requires 16-byte alignment (4 floats)
          if (ko + c_a + 4 <= K && (reinterpret_cast<uintptr_t>(src_a) & 15) == 0) {
              cp_async_16(sp_a, src_a);
          } else {
              #pragma unroll
              for(int e=0; e<4; e++) {
                  sp_a[e] = (ko + c_a + e < K) ? src_a[e] : 0.0f;
              }
          }
      } else {
          *(float4*)sp_a = {0.0f, 0.0f, 0.0f, 0.0f};
      }

      int r_b = tid / 32, c_b = (tid % 32) * 4;
      float* sp_b = &Bs[buf_idx * BK * BN + r_b * BN + c_b];
      if (ko + r_b < K && bx * BN + c_b < N) {
          const float* src_b = &dB[(ko + r_b) * ldb + c_b];
          // cp_async_16 requires 16-byte alignment (4 floats)
          if (bx * BN + c_b + 4 <= N && (reinterpret_cast<uintptr_t>(src_b) & 15) == 0) {
              cp_async_16(sp_b, src_b);
          } else {
              #pragma unroll
              for(int e=0; e<4; e++) {
                  sp_b[e] = (bx * BN + c_b + e < N) ? src_b[e] : 0.0f;
              }
          }
      } else {
          *(float4*)sp_b = {0.0f, 0.0f, 0.0f, 0.0f};
      }
      asm volatile("cp.async.commit_group;\n");
  };

  int write_idx = 0;
  load_tile(0, write_idx);
  
  for (int k = 0; k < K; k += BK) {
      asm volatile("cp.async.wait_group 0;\n");
      __syncthreads();
      
      int read_idx = write_idx;
      write_idx = 1 - write_idx;
      if (k + BK < K) load_tile(k + BK, write_idx);

      // Dot product loop with float4 reading and unrolling
      #pragma unroll
      for (int d = 0; d < BK; d++) {
          float4 regM_v[2];
          float4 regN_v;

          // As is [row][col] where r=0..127, c=0..15.
          // regM: A[tRow*8+0..7][d]. 
          // This is NOT a float4 load in BK dimension.
          // But we can manually load A elements.
          float regM[TM];
          #pragma unroll
          for(int i=0; i<TM; i++) regM[i] = As[read_idx * BM * BK + (tRow * TM + i) * BK + d];

          // Bs is [row][col] where r=0..15, c=0..127.
          // regN: B[d][tCol*4+0..3]. 
          // THIS IS A FLOAT4 LOAD!
          regN_v = *(float4*)&Bs[read_idx * BK * BN + d * BN + tCol * TN];

          float* rN = (float*)&regN_v;
          #pragma unroll
          for (int i = 0; i < TM; i++) {
              #pragma unroll
              for (int j = 0; j < TN; j++) {
                  results[i][j] += regM[i] * rN[j];
              }
          }
      }
  }

  #pragma unroll
  for (int i = 0; i < TM; i++) {
      int rb = tRow * TM + i;
      int r = by * BM + rb;
      if (r < M) {
          #pragma unroll
          for (int j = 0; j < TN; j++) {
              int cb = tCol * TN + j;
              int c = bx * BN + cb;
              if (c < N) {
                 float val = results[i][j];
                 float* C_ptr = dC + rb * ldc + cb;
                 if (beta != 0.0f) *C_ptr = alpha * val + beta * (*C_ptr);
                 else *C_ptr = alpha * val;
              }
          }
      }
  }
}

extern "C" void mycublasSgemm(
  mycublasHandle_t handle,
  int M, int N, int K,
  float alpha,
  const float *A, int lda,
  const float *B, int ldb,
  float beta,
  float *C, int ldc)
{
  cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
  size_t sz = 2 * (BM * BK + BK * BN) * sizeof(float);
  dim3 block(512);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_optimized_v9_kernel<<<grid, block, sz, stream>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
