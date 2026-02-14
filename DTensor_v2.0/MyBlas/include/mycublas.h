#ifndef MYCUBLAS_H
#define MYCUBLAS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Internal context definition
struct mycublasContext {
    int deviceId;
    cudaStream_t stream;
};
typedef struct mycublasContext *mycublasHandle_t;

// Status codes (simplified)
typedef enum {
    MYCUBLAS_STATUS_SUCCESS = 0,
    MYCUBLAS_STATUS_NOT_INITIALIZED = 1,
    MYCUBLAS_STATUS_ALLOC_FAILED = 2,
    MYCUBLAS_STATUS_INVALID_VALUE = 3,
    MYCUBLAS_STATUS_ARCH_MISMATCH = 4,
    MYCUBLAS_STATUS_MAPPING_ERROR = 5,
    MYCUBLAS_STATUS_EXECUTION_FAILED = 6,
    MYCUBLAS_STATUS_INTERNAL_ERROR = 7,
    MYCUBLAS_STATUS_NOT_SUPPORTED = 8,
    MYCUBLAS_STATUS_LICENSE_ERROR = 9
} mycublasStatus_t;

// Management functions
mycublasStatus_t mycublasCreate(mycublasHandle_t *handle);
mycublasStatus_t mycublasDestroy(mycublasHandle_t handle);
mycublasStatus_t mycublasSetStream(mycublasHandle_t handle, cudaStream_t streamId);

// Level 3 BLAS
void mycublasSgemm(
    mycublasHandle_t handle,
    int M, int N, int K, 
    float alpha, 
    const float *d_A, int lda, 
    const float *d_B, int ldb, 
    float beta, 
    float *d_C, int ldc
);

// Tensor Core (TF32) - Ampere+ architecture
// Automatically uses TF32 Tensor Cores on Ampere+ GPUs (CC >= 8.0)
// Falls back to CUDA cores on older architectures
void mycublasSgemm_TensorCore(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const float *d_A, int lda,
    const float *d_B, int ldb,
    float beta,
    float *d_C, int ldc
);

// Tensor Core (TF32) with explicit control
// use_tf32: true to force TF32 (if supported), false to use CUDA cores
void mycublasSgemm_TensorCore_Explicit(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const float *d_A, int lda,
    const float *d_B, int ldb,
    float beta,
    float *d_C, int ldc,
    bool use_tf32
);

// Tensor Core (FP16)
// A: FP16, B: FP16, C: FP16 (accumulate in FP32 usually, but output is FP16 here for simplicity or as requested)
// We will assume standard Hgemm: C = alpha * A * B + beta * C
void mycublasHgemm(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __half alpha,
    const __half *d_A, int lda,
    const __half *d_B, int ldb,
    const __half beta,
    __half *d_C, int ldc
);

// Tensor Core (BF16)
// A: BF16, B: BF16, C: BF16 (Accumulate in FP32 usually)
void mycublasBgemm(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __nv_bfloat16 alpha,
    const __nv_bfloat16 *d_A, int lda,
    const __nv_bfloat16 *d_B, int ldb,
    const __nv_bfloat16 beta,
    __nv_bfloat16 *d_C, int ldc
);

// Tensor Core (FP64 / DMMA)
// A, B, C: Double
void mycublasDgemm(
    mycublasHandle_t handle,
    int M, int N, int K,
    const double alpha,
    const double *d_A, int lda,
    const double *d_B, int ldb,
    const double beta,
    double *d_C, int ldc
);

// Strided Batched GEMM
void mycublasSgemmStridedBatched(
    mycublasHandle_t handle,
    int M, int N, int K,
    const float alpha,
    const float *d_A, int lda, long long int strideA,
    const float *d_B, int ldb, long long int strideB,
    const float beta,
    float *d_C, int ldc, long long int strideC,
    int batchCount
);

void mycublasHgemmStridedBatched(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __half alpha,
    const __half *d_A, int lda, long long int strideA,
    const __half *d_B, int ldb, long long int strideB,
    const __half beta,
    __half *d_C, int ldc, long long int strideC,
    int batchCount
);

// V2: High-performance from-scratch implementation (256x128x32, 4-stage pipeline)
void mycublasHgemmStridedBatchedV2(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __half alpha,
    const __half *d_A, int lda, long long int strideA,
    const __half *d_B, int ldb, long long int strideB,
    const __half beta,
    __half *d_C, int ldc, long long int strideC,
    int batchCount
);

void mycublasBgemmStridedBatched(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __nv_bfloat16 alpha,
    const __nv_bfloat16 *d_A, int lda, long long int strideA,
    const __nv_bfloat16 *d_B, int ldb, long long int strideB,
    const __nv_bfloat16 beta,
    __nv_bfloat16 *d_C, int ldc, long long int strideC,
    int batchCount
);

void mycublasDgemmStridedBatched(
    mycublasHandle_t handle,
    int M, int N, int K,
    const double alpha,
    const double *d_A, int lda, long long int strideA,
    const double *d_B, int ldb, long long int strideB,
    const double beta,
    double *d_C, int ldc, long long int strideC,
    int batchCount
);

// Tensor Core (Int8 / IMMA)
// A, B: int8_t, C: int32_t (Accumulator)
void mycublasIgemm(
    mycublasHandle_t handle,
    int M, int N, int K,
    const int32_t alpha,
    const int8_t *d_A, int lda,
    const int8_t *d_B, int ldb,
    const int32_t beta,
    int32_t *d_C, int ldc
);

#ifdef __cplusplus
}
#endif

#endif // MYCUBLAS_H