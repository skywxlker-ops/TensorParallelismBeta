#include "dnn/FusedLMHeadKernel.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

namespace OwnTensor {
namespace dnn {

// ---------------------------------------------------------------------------
// Persistent cuBLAS handle (created once per process)
// ---------------------------------------------------------------------------
static cublasHandle_t g_handle = nullptr;
static bool           g_tf32_set = false;

static cublasHandle_t get_handle() {
    if (!g_handle) {
        cublasStatus_t s = cublasCreate(&g_handle);
        if (s != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[FusedLMHead] cublasCreate failed: %d\n", (int)s);
            exit(1);
        }
    }
    if (!g_tf32_set) {
        // Allow TF32 Tensor Cores on Ampere+ for FP32 GEMMs.
        // cublasSetMathMode selects CUBLAS_TF32_TENSOR_OP_MATH which maps to
        // CUBLAS_COMPUTE_32F_FAST_TF32 internally.
        cublasSetMathMode(g_handle, CUBLAS_TF32_TENSOR_OP_MATH);
        g_tf32_set = true;
    }
    return g_handle;
}

// ---------------------------------------------------------------------------
// forward:  logits[BT, vocab] = hidden[BT, K] @ weight.T[K, vocab]
//
// Row-major A[M,K] * B[N,K]^T = C[M,N]:
//   cublasSgemm(OP_T, OP_N, N, M, K, alpha, B, K, A, K, beta, C, N)
//
// With TF32 math mode active this uses Tensor Cores on sm>=80.
// ---------------------------------------------------------------------------
void fused_lm_head_forward(
    const float* hidden,
    const float* weight,
    float*       logits,
    int BT,
    int n_embd,
    int vocab_size)
{
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasSgemm(
        get_handle(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        vocab_size, BT, n_embd,
        &alpha,
        weight,  n_embd,
        hidden,  n_embd,
        &beta,
        logits,  vocab_size);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[FusedLMHead] fwd sgemm failed: %d\n", (int)s);
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// backward:
//   d_hidden[BT, K]      = d_logits[BT, V] @ weight[V, K]
//   d_weight[V, K]       = d_logits.T[V, BT] @ hidden[BT, K]
//
// d_hidden = d_logits[BT,V] @ weight[V,K]:
//   Row-major A[M,K2] @ B[K2,N]  where M=BT, N=K, K2=V
//   cublasSgemm(OP_N, OP_N, N, M, K2, alpha, B, N, A, K2, beta, C, N)
//   = cublasSgemm(OP_N, OP_N, n_embd, BT, vocab, 1, weight, n_embd,
//                 d_logits, vocab, 0, d_hidden, n_embd)
//
// d_weight = d_logits^T[V,BT] @ hidden[BT,K]:
//   Row-major A^T[V,BT] @ B[BT,K]  where M=V, N=K, K2=BT
//   cublasSgemm(OP_N, OP_T, N, M, K2, alpha, B, N, A, M, beta, C, N)
//   = cublasSgemm(OP_N, OP_T, n_embd, vocab, BT, 1, hidden, n_embd,
//                 d_logits, vocab, 1, d_weight, n_embd)
//   Note: beta=1 so d_weight is accumulated, not overwritten.
// ---------------------------------------------------------------------------
void fused_lm_head_backward(
    const float* d_logits,
    const float* hidden,
    const float* weight,
    float*       d_hidden,
    float*       d_weight,
    int BT,
    int n_embd,
    int vocab_size)
{
    cublasHandle_t handle = get_handle();
    float alpha = 1.0f, beta0 = 0.0f, beta1 = 1.0f;

    // d_hidden
    cublasStatus_t s1 = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n_embd, BT, vocab_size,
        &alpha,
        weight,   n_embd,
        d_logits, vocab_size,
        &beta0,
        d_hidden, n_embd);
    if (s1 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[FusedLMHead] bwd d_hidden sgemm failed: %d\n", (int)s1);
        exit(1);
    }

    // d_weight (accumulated: beta=1)
    cublasStatus_t s2 = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n_embd, vocab_size, BT,
        &alpha,
        hidden,   n_embd,
        d_logits, vocab_size,
        &beta1,
        d_weight, n_embd);
    if (s2 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[FusedLMHead] bwd d_weight sgemm failed: %d\n", (int)s2);
        exit(1);
    }
}

} // namespace dnn
} // namespace OwnTensor
