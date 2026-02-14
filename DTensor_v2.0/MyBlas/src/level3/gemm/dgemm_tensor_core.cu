#include "mycublas.h"
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// FP64 WMMA/DMMA KERNEL
// Tile: 64x64x8
// Threads: 256
// ============================================================================

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8; // Small K due to double precision DMMA constraints
// DMMA shape is 8x8x4.
// Threads: 256.
// Warp tile size?
// 256 threads = 8 warps.
// Grid 64x64.
// Warp can likely handle 16x32 or 32x16?
// If we use simple double buffering with no fancy warp tiling (as reference did mostly simpler logic for FP64? No, reference used reg tiling).
// Reference Code:
/*
   double res[16] = {0.0}, rM[4], rN[4];
   for (int bk = 0; bk < K; bk += BK) {
      load tiles...
      for (int d = 0; d < BK; d++) {
         for (int i = 0; i < 4; i++) rM[i] = As[(tR*4+i)*BK+d];
         for (int i = 0; i < 4; i++) rN[i] = Bs[d*BN+tC*4+i];
         ... outer product
      }
   }
*/
// Reference code uses MANUAL register tiling (Volta/CUDA Core style) for FP64 instead of DMMA in that specific block?
// Wait, "matmul_fp64_optimized" in reference has NO `wmma::` calls. 
// It uses shared memory and FMA ops (`res += rM * rN`).
// This means the reference FP64 kernel provided is a CUDA Core optimization, NOT Tensor Core DMMA.
// That explains `BK=8`.
// However, I previously implemented `dgemm_tensor_core.cu` utilizing actual DMMA (`wmma::mma_sync`).
// DMMA provides huge speedups on A100.
// Should I stick to my DMMA implementation or use the user's "Optimized Reference"?
// The user asked "can we optimize ... using the following code as a reference".
// The provided reference FP64 kernel is likely slower than DMMA on A100/H100 because it doesn't use tensor cores.
// But the user *provided* it as the "optimized" version.
// Maybe I should provide a Hybrid?
// Or maybe I should investigate why they prefer this?
// Ah, look at reference logic again: `matmul_fp64_optimized`.
// No `wmma` header included or used in that function.
// But my goal is "Optimizing BLAS Library" targeting "A100, H100".
// On A100, DMMA is the way.
// My previous tasks built `dgemm_tensor_core.cu` WITH DMMA.
// If I replace it with this CUDA Core version, I might regress performance on A100.
// BUT, the existing DMMA implementation I made was likely basic (no tiling/double buffering).
// I should Implement a **DMMA version** that uses the Tiling/Double Buffering *structure* from the reference, but keeps `wmma` instructions.
// Structure:
// Shared Mem As[2][BM][BK], Bs[2][BK][BN].
// Pipeline load & compute.
// Use `wmma::load_matrix_sync` and `wmma::mma_sync`.

// DMMA Config:
// Shape: 8x8x4.
// Block: 64x64? 64/8 = 8 tiles wide.
// Warps: 256 threads = 8 warps.
// 8 warps for 64x64 output?
// 64x64 = 4096 elements.
// Each warp handles 512 elements (32x16 or 16x32).
// (16x32) * 8 warps = 128x32 ... layout is tricky.
// Let's try 4 warps x 2 warps grid = 8 warps.
// Warp Tile: 16x32.
// 4 * 16 = 64 (M).
// 2 * 32 = 64 (N).
// Matches.

constexpr int WM = 16;
constexpr int WN = 32;
constexpr int DMMA_M = 8;
constexpr int DMMA_N = 8;
constexpr int DMMA_K = 4;
constexpr int PAD = 4; // Double alignment

__global__ void dgemm_optimized_kernel(
    int M, int N, int K,
    double alpha,
    const double* __restrict__ A, int lda,
    const double* __restrict__ B, int ldb,
    double beta,
    double* __restrict__ C, int ldc)
{
    // Indexes
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / 2; // 0..3 (4 rows of warps)
    const int warp_col = warp_id % 2; // 0..1 (2 cols of warps)
    
    // Pointers
    // A: Row Major.
    const double *Ap = A;
    const double *Bp = B;
    double *Cp = C;
    
    // Shared Memory
    // As[2][BM][BK + PAD]
    // BM=64, BK=16 (needs to be multiple of DMMA_K=4. Larger BK better for reuse).
    // Let's try BK=16. 
    // Size: 2 * 64 * 16 * 8 bytes = 16KB. Very small. OK.
    const int MY_BK = 16;
    
    __shared__ double As[2][BM][MY_BK + PAD];
    __shared__ double Bs[2][MY_BK][BN + PAD];
    
    // Fragments
    // WarpTile 16x32.
    // DMMA 8x8x4.
    // M dim: 16 / 8 = 2.
    // N dim: 32 / 8 = 4.
    // Total 2x4 = 8 mma ops.
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> af[2][4]; // dim K (buffer depth for loop)? No, tiling spatial.
    // Wait, matrix_a fragment matches mma_sync shape. 
    // We need 'a' fragments for M dimension (2 of them).
    // range i: 0..1.
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frags[2];

    // we need 'b' fragments for N dimension (4 of them).
    // range j: 0..3.
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frags[4];

    // accumulators [2][4]
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc[2][4];

    // Init Acc
    #pragma unroll
    for(int i=0; i<2; i++) 
        for(int j=0; j<4; j++) 
            wmma::fill_fragment(acc[i][j], 0.0);

    // Load Pipeline
    auto load_tiles = [&](int ko, int idx) {
        // Load A: BM x BK (64 x 16). 1024 elements.
        // Threads 256. 4 elems/thread.
        for (int i = tid; i < BM * MY_BK; i += 256) {
            int r = i / MY_BK;
            int c = i % MY_BK;
            int gr = by * BM + r;
            int gc = ko + c;
            As[idx][r][c] = (gr < M && gc < K) ? Ap[gr*lda + gc] : 0.0;
        }
        // Load B: BK x BN (16 x 64). 1024 elements.
        for (int i = tid; i < MY_BK * BN; i += 256) {
            int r = i / BN;
            int c = i % BN;
            int gr = ko + r;
            int gc = bx * BN + c;
            Bs[idx][r][c] = (gr < K && gc < N) ? Bp[gr*ldb + gc] : 0.0;
        }
    };

    int write_idx = 0;
    load_tiles(0, write_idx);
    __syncthreads();
    
    for (int k = 0; k < K; k += MY_BK) {
        int read_idx = write_idx;
        write_idx = 1 - write_idx;
        if (k + MY_BK < K) load_tiles(k + MY_BK, write_idx);
        
        // Compute Loop over BK
        // Step size DMMA_K = 4.
        // MY_BK = 16. So 4 steps.
        #pragma unroll
        for (int ks = 0; ks < MY_BK; ks += 4) {
            
            // Load A frags for WarpRows
            // Warp covers rows: warp_row * WM (16) ... +15.
            // i=0 -> rows 0..7 local. i=1 -> rows 8..15 local.
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                wmma::load_matrix_sync(a_frags[i], &As[read_idx][warp_row * WM + i * 8][ks], MY_BK + PAD);
            }
            
            // Load B frags for WarpCols
            // Warp covers cols: warp_col * WN (32) ... +31.
            // j=0..3 -> cols 0,8,16,24 offsets.
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::load_matrix_sync(b_frags[j], &Bs[read_idx][ks][warp_col * WN + j * 8], BN + PAD);
            }
            
            // Math
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    wmma::mma_sync(acc[i][j], a_frags[i], b_frags[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    
    // Store
    // 256 threads. Each warp stores 16x32 tile.
    // Reuse As?
    // As size doubles = 16KB.
    // Need buffer for 16x32 doubles = 512 doubles.
    // 8 warps * 512 = 4096 doubles.
    // 4096 * 8 bytes = 32KB.
    // As is 16KB. Bs is 16KB. Total 32KB.
    // We can reuse As + Bs or re-declare dynamic shared mem.
    // Or just store directly since 64x64 is small?
    // Direct store with wmma::store_matrix_sync to global might be slow due to striding?
    // Let's try direct store first, if buggy/slow we use SMEM.
    
    // wait, store_matrix_sync requires strict layout.
    // Generic C pointer often has large ldc.
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            // Global coords
            int r = by*BM + warp_row*WM + i*8;
            int c = bx*BN + warp_col*WN + j*8;
            
            // Use pointer arithmetic to point to C[r][c]
            // Bounds check?
            if (r < M && c < N) {
                // wmma store
                // NOTE: standard store overwrites. We need accumulate.
                // Load C fragment, add, store?
                // wmma::load/store for C is possible.
                // But beta handling is cleaner element-wise.
                // Let's store to SMEM and write out manual.
                // We reuse As unioned with Bs?
                // Shared mem pointer alias.
                double* sm = reinterpret_cast<double*>(As);
                // Warp offset: warp_id * 512.
                // 8 warps * 512 = 4096. 
                // As+Bs capacity = 1024 + 1024 = 2048 doubles. 
                // NOT ENOUGH.
                // We need to tile the store or use atomic add? No.
                // Just do standard read-modify-write per thread mapping?
                // store_matrix_sync to Global is supported.
                // Does it support beta? No.
                // So we must read C, compute, write C.
                // Doing this fragment-wise is inefficient.
                // Let's optimize later. For now, correctness.
                // We can't easily coalesce without SMEM.
                // Let's serialize warps if needed? 
                // No, just define a smaller SMEM buffer loop?
                // Or: Increase SMEM size? 32KB fits on almost all GPUs.
                // `__shared__ double As[2][BM][MY_BK+PAD]` -> As[2][64][20] = 2560.
                // Let's just increase buffer size for store specifically.
                // Union?
            }
        }
    }
    
    // Re-declare SMEM for Output (Union logic manually)
    // We need 4096 doubles. 
    // Existing As+Bs = 2*64*20 + 2*16*68 = 2560 + 2176 = 4736 doubles.
    // Fits! 
    double* sm_out = reinterpret_cast<double*>(As);
    double* warp_sm = sm_out + warp_id * 512;
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
             // Store 8x8 tile to linear SMEM
             // stride 8 or 32?
             // Let's store compact 8x8 output.
             // destination in warp_sm: (i*8)*32 + (j*8) start.
             // stride 32 (WN).
             wmma::store_matrix_sync(warp_sm + (i*8)*32 + (j*8), acc[i][j], 32, wmma::mem_row_major);
        }
    }
    __syncthreads();
    // No syncthreads needed between warps if disjoint regions.
    
    // Now write out warp_sm (16x32) to Global
    // 512 elements. 32 threads. 16 elems/thread.
    int gr_start = by*BM + warp_row*WM;
    int gc_start = bx*BN + warp_col*WN;
    
    for (int t = 0; t < 16; t++) {
        int tid_in_warp = tid % 32;
        int idx = tid_in_warp + t * 32; // 0..511
        // Map idx to row/col in 16x32 tile
        int r = idx / 32;
        int c = idx % 32;
        
        int gr = gr_start + r;
        int gc = gc_start + c;
        
        if (gr < M && gc < N) {
            double val = warp_sm[idx];
            double c_val = Cp[gr * ldc + gc];
            Cp[gr * ldc + gc] = alpha * val + beta * c_val;
        }
    }
}

extern "C" void mycublasDgemm(
    mycublasHandle_t handle,
    int M, int N, int K,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double beta,
    double *C, int ldc)
{
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dgemm_optimized_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
