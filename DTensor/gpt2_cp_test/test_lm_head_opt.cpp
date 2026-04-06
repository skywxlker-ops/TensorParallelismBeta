// =============================================================================
// test_lm_head_opt.cpp
//
// Correctness + timing test for FusedLMHeadKernel (TF32 Tensor Core path).
//
// RED:   References fused_lm_head_forward / fused_lm_head_backward which do
//        not exist until FusedLMHeadKernel.cu is compiled.
//
// GREEN: All PASS checks hold when the implementation is complete.
//
// The LM head computes:
//   forward:  logits[BT, vocab] = hidden[BT, n_embd] @ weight.T[n_embd, vocab]
//   backward: d_hidden[BT, n_embd] = d_logits[BT, vocab] @ weight[vocab, n_embd]
//             d_weight[vocab, n_embd] = d_logits.T[vocab, BT] @ hidden[BT, n_embd]
//
// Correctness tolerance:
//   TF32 Tensor Core: max|result - fp32_ref| < 0.1  (TF32 rounds mantissa to 10 bits)
//
// Shapes tested:
//   BT=2048  n_embd=384   vocab=50304  -- GPT-2 small, B=4 T=512
//   BT=512   n_embd=384   vocab=50304  -- smaller batch for timing
//
// Build:  make test_lm_head_opt
// Run:    ./test_lm_head_opt_exec
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "dnn/FusedLMHeadKernel.h"

using namespace OwnTensor::dnn;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}
static void cublas_check(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error (%s): %d\n", msg, (int)s);
        exit(1);
    }
}

static uint32_t g_rng = 0x12345678u;
static float lcg_randf() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (static_cast<float>(g_rng >> 8) / static_cast<float>(1 << 24)) - 0.5f;
}

static float* alloc_rand_f32(int64_t n, float scale = 1.0f) {
    std::vector<float> h(n);
    for (auto& v : h) v = lcg_randf() * scale;
    float* d = nullptr;
    cuda_check(cudaMalloc(&d, n * sizeof(float)), "alloc");
    cuda_check(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice), "H2D");
    return d;
}

static float* alloc_zeros_f32(int64_t n) {
    float* d = nullptr;
    cuda_check(cudaMalloc(&d, n * sizeof(float)), "alloc zeros");
    cuda_check(cudaMemset(d, 0, n * sizeof(float)), "memset");
    return d;
}

static std::vector<float> d2h(const float* d, int64_t n) {
    std::vector<float> h(n);
    cuda_check(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
    return h;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
        mx = std::max(mx, std::abs(a[i] - b[i]));
    return mx;
}

static bool report(const char* label, float diff, float tol) {
    bool ok = diff < tol;
    printf("  [%-40s] diff=%.4f  tol=%.4f  %s\n", label, diff, tol, ok ? "PASS" : "FAIL");
    return ok;
}

// Reference: FP32 cuBLAS (no Tensor Cores) using standard cublasSgemm
// logits[BT, vocab] = hidden[BT, K] @ weight.T[K, vocab]
// Row-major: cublas(OP_T, OP_N, vocab, BT, K, 1, weight, K, hidden, K, 0, logits, vocab)
static void ref_lm_head_forward(
    cublasHandle_t handle,
    const float* hidden, const float* weight, float* logits,
    int BT, int n_embd, int vocab_size)
{
    float alpha = 1.0f, beta = 0.0f;
    cublas_check(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        vocab_size, BT, n_embd,
        &alpha,
        weight, n_embd,
        hidden, n_embd,
        &beta,
        logits, vocab_size
    ), "ref forward sgemm");
}

// Reference backward dW = dlogits.T @ hidden   and   dhidden = dlogits @ weight
static void ref_lm_head_backward(
    cublasHandle_t handle,
    const float* d_logits, const float* hidden, const float* weight,
    float* d_hidden, float* d_weight,
    int BT, int n_embd, int vocab_size)
{
    float alpha = 1.0f, beta = 0.0f;
    // d_hidden[BT, n_embd] = d_logits[BT, vocab] @ weight[vocab, n_embd]
    // row-major: cublas(OP_N, OP_N, n_embd, BT, vocab, ...)
    cublas_check(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n_embd, BT, vocab_size,
        &alpha,
        weight,   n_embd,
        d_logits, vocab_size,
        &beta,
        d_hidden, n_embd
    ), "ref backward dhidden");

    // d_weight[vocab, n_embd] = d_logits.T[vocab, BT] @ hidden[BT, n_embd]
    // row-major: cublas(OP_N, OP_T, n_embd, vocab, BT, ...)
    cublas_check(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n_embd, vocab_size, BT,
        &alpha,
        hidden,   n_embd,
        d_logits, vocab_size,
        &beta,
        d_weight, n_embd
    ), "ref backward dweight");
}

// ---------------------------------------------------------------------------
// Per-shape test
// ---------------------------------------------------------------------------
static bool test_shape(cublasHandle_t handle, int BT, int n_embd, int vocab_size) {
    printf("  Shape BT=%d  n_embd=%d  vocab=%d\n", BT, n_embd, vocab_size);
    bool ok = true;

    g_rng += (uint32_t)(BT + n_embd);
    float* d_hidden  = alloc_rand_f32((int64_t)BT * n_embd, 0.1f);
    float* d_weight  = alloc_rand_f32((int64_t)vocab_size * n_embd, 0.02f);
    float* d_dlogits = alloc_rand_f32((int64_t)BT * vocab_size, 0.01f);

    // Reference forward
    float* d_logits_ref = alloc_zeros_f32((int64_t)BT * vocab_size);
    ref_lm_head_forward(handle, d_hidden, d_weight, d_logits_ref, BT, n_embd, vocab_size);
    cuda_check(cudaDeviceSynchronize(), "ref fwd sync");

    // Fused forward (TF32 Tensor Core)
    float* d_logits_tc = alloc_zeros_f32((int64_t)BT * vocab_size);
    fused_lm_head_forward(d_hidden, d_weight, d_logits_tc, BT, n_embd, vocab_size);
    cuda_check(cudaDeviceSynchronize(), "tc fwd sync");

    auto h_ref = d2h(d_logits_ref, (int64_t)BT * vocab_size);
    auto h_tc  = d2h(d_logits_tc,  (int64_t)BT * vocab_size);
    ok &= report("fwd logits (TF32 vs FP32 ref)", max_abs_diff(h_ref, h_tc), 0.1f);

    // Reference backward
    float* d_dhidden_ref = alloc_zeros_f32((int64_t)BT * n_embd);
    float* d_dweight_ref = alloc_zeros_f32((int64_t)vocab_size * n_embd);
    ref_lm_head_backward(handle, d_dlogits, d_hidden, d_weight,
                         d_dhidden_ref, d_dweight_ref, BT, n_embd, vocab_size);
    cuda_check(cudaDeviceSynchronize(), "ref bwd sync");

    // Fused backward
    float* d_dhidden_tc = alloc_zeros_f32((int64_t)BT * n_embd);
    float* d_dweight_tc = alloc_zeros_f32((int64_t)vocab_size * n_embd);
    fused_lm_head_backward(d_dlogits, d_hidden, d_weight,
                           d_dhidden_tc, d_dweight_tc, BT, n_embd, vocab_size);
    cuda_check(cudaDeviceSynchronize(), "tc bwd sync");

    auto hd_ref = d2h(d_dhidden_ref, (int64_t)BT * n_embd);
    auto hd_tc  = d2h(d_dhidden_tc,  (int64_t)BT * n_embd);
    ok &= report("bwd dhidden (TF32 vs FP32 ref)", max_abs_diff(hd_ref, hd_tc), 0.1f);

    auto hw_ref = d2h(d_dweight_ref, (int64_t)vocab_size * n_embd);
    auto hw_tc  = d2h(d_dweight_tc,  (int64_t)vocab_size * n_embd);
    ok &= report("bwd dweight (TF32 vs FP32 ref)", max_abs_diff(hw_ref, hw_tc), 0.1f);

    cudaFree(d_hidden); cudaFree(d_weight); cudaFree(d_dlogits);
    cudaFree(d_logits_ref); cudaFree(d_logits_tc);
    cudaFree(d_dhidden_ref); cudaFree(d_dweight_ref);
    cudaFree(d_dhidden_tc);  cudaFree(d_dweight_tc);
    return ok;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
static void timing(cublasHandle_t handle, int BT, int n_embd, int vocab_size, int NITER) {
    printf("\n--- Timing BT=%d n_embd=%d vocab=%d, %d iters ---\n",
           BT, n_embd, vocab_size, NITER);
    g_rng += 77777u;
    float* d_hidden   = alloc_rand_f32((int64_t)BT * n_embd, 0.1f);
    float* d_weight   = alloc_rand_f32((int64_t)vocab_size * n_embd, 0.02f);
    float* d_dlogits  = alloc_rand_f32((int64_t)BT * vocab_size, 0.01f);
    float* d_logits   = alloc_zeros_f32((int64_t)BT * vocab_size);
    float* d_dhidden  = alloc_zeros_f32((int64_t)BT * n_embd);
    float* d_dweight  = alloc_zeros_f32((int64_t)vocab_size * n_embd);

    // Warm-up
    for (int w = 0; w < 3; w++) {
        ref_lm_head_forward(handle, d_hidden, d_weight, d_logits, BT, n_embd, vocab_size);
        fused_lm_head_forward(d_hidden, d_weight, d_logits, BT, n_embd, vocab_size);
    }
    cudaDeviceSynchronize();

    // Reference forward
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NITER; i++)
        ref_lm_head_forward(handle, d_hidden, d_weight, d_logits, BT, n_embd, vocab_size);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_ref_fwd = std::chrono::duration<double, std::milli>(t1 - t0).count() / NITER;

    // Fused forward
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NITER; i++)
        fused_lm_head_forward(d_hidden, d_weight, d_logits, BT, n_embd, vocab_size);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_tc_fwd = std::chrono::duration<double, std::milli>(t3 - t2).count() / NITER;

    printf("  Forward   FP32 cuBLAS: %6.3f ms   TF32 TC: %6.3f ms   speedup: %.2fx\n",
           ms_ref_fwd, ms_tc_fwd, ms_ref_fwd / ms_tc_fwd);

    // Reference backward
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NITER; i++)
        ref_lm_head_backward(handle, d_dlogits, d_hidden, d_weight,
                             d_dhidden, d_dweight, BT, n_embd, vocab_size);
    cudaDeviceSynchronize();
    auto t5 = std::chrono::high_resolution_clock::now();
    double ms_ref_bwd = std::chrono::duration<double, std::milli>(t5 - t4).count() / NITER;

    // Fused backward
    auto t6 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NITER; i++)
        fused_lm_head_backward(d_dlogits, d_hidden, d_weight,
                               d_dhidden, d_dweight, BT, n_embd, vocab_size);
    cudaDeviceSynchronize();
    auto t7 = std::chrono::high_resolution_clock::now();
    double ms_tc_bwd = std::chrono::duration<double, std::milli>(t7 - t6).count() / NITER;

    printf("  Backward  FP32 cuBLAS: %6.3f ms   TF32 TC: %6.3f ms   speedup: %.2fx\n",
           ms_ref_bwd, ms_tc_bwd, ms_ref_bwd / ms_tc_bwd);

    cudaFree(d_hidden); cudaFree(d_weight); cudaFree(d_dlogits);
    cudaFree(d_logits); cudaFree(d_dhidden); cudaFree(d_dweight);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    cuda_check(cudaSetDevice(0), "setDevice");

    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");

    printf("\n=== test_lm_head_opt ===\n\n");
    bool all_ok = true;

    printf("--- Correctness ---\n");
    all_ok &= test_shape(handle, 2048, 384, 50304);   // B=4 T=512 GPT-2 small
    all_ok &= test_shape(handle, 512,  384, 50304);   // smaller
    printf("\n");

    // Timing
    timing(handle, 2048, 384, 50304, 50);
    timing(handle, 512,  384, 50304, 50);

    cublasDestroy(handle);

    printf("\n=== %s ===\n\n", all_ok ? "ALL PASS" : "SOME FAIL");
    return all_ok ? 0 : 1;
}
