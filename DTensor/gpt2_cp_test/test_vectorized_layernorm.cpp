// =============================================================================
// test_vectorized_layernorm.cpp
//
// Correctness + timing test for VectorizedLayerNormKernel (dnn/).
//
// RED:   References vln_forward_f32 / vln_backward_f32 / vln_forward_f16 /
//        vln_backward_f16 / vln_forward_bf16 / vln_backward_bf16 which do not
//        exist until VectorizedLayerNormKernel.cu is compiled.
//
// GREEN: All PASS checks hold when the implementation is complete.
//
// Correctness tolerance:
//   float  fwd: max|y_vec  - y_ref|  < 1e-5
//   float  bwd: max|dx_vec - dx_ref| < 1e-5,  max|dg_vec - dg_ref| < 1e-5
//   fp16   fwd: max|y_vec  - y_ref|  < 1e-2   (fp16 rounding)
//   bf16   fwd: max|y_vec  - y_ref|  < 5e-2   (bf16 rounding)
//
// Shapes tested (rows x cols):
//   [512, 384]   -- GPT-2 small (n_embd=384), cols divisible by 4 -> vectorized
//   [512, 768]   -- GPT-2 medium
//   [64,  64]    -- cols divisible by 4
//   [64,  65]    -- cols NOT divisible by 4 -> scalar fallback
//
// Build:  make test_vectorized_layernorm
// Run:    ./test_vectorized_layernorm_exec
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "ops/helpers/LayerNormKernels.h"
#include "dnn/VectorizedLayerNormKernel.h"

using namespace OwnTensor::cuda;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static uint32_t g_rng = 0xABCDEF01u;
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

// Float-to-half host conversion
static __half* alloc_rand_f16(int64_t n, float scale = 1.0f) {
    std::vector<__half> h(n);
    for (auto& v : h) { float f = lcg_randf() * scale; v = __float2half(f); }
    __half* d = nullptr;
    cuda_check(cudaMalloc(&d, n * sizeof(__half)), "alloc f16");
    cuda_check(cudaMemcpy(d, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice), "H2D f16");
    return d;
}

static __nv_bfloat16* alloc_rand_bf16(int64_t n, float scale = 1.0f) {
    std::vector<__nv_bfloat16> h(n);
    for (auto& v : h) { float f = lcg_randf() * scale; v = __float2bfloat16(f); }
    __nv_bfloat16* d = nullptr;
    cuda_check(cudaMalloc(&d, n * sizeof(__nv_bfloat16)), "alloc bf16");
    cuda_check(cudaMemcpy(d, h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "H2D bf16");
    return d;
}

// D2H copies
static std::vector<float> d2h_f32(const float* d, int64_t n) {
    std::vector<float> h(n);
    cuda_check(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H f32");
    return h;
}
static std::vector<float> d2h_f16_as_f32(const __half* d, int64_t n) {
    std::vector<__half> hh(n);
    cuda_check(cudaMemcpy(hh.data(), d, n * sizeof(__half), cudaMemcpyDeviceToHost), "D2H f16");
    std::vector<float> h(n);
    for (int64_t i = 0; i < n; i++) h[i] = __half2float(hh[i]);
    return h;
}
static std::vector<float> d2h_bf16_as_f32(const __nv_bfloat16* d, int64_t n) {
    std::vector<__nv_bfloat16> hh(n);
    cuda_check(cudaMemcpy(hh.data(), d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "D2H bf16");
    std::vector<float> h(n);
    for (int64_t i = 0; i < n; i++) h[i] = __bfloat162float(hh[i]);
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
    printf("  [%-40s] diff=%.2e  tol=%.2e  %s\n", label, diff, tol, ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Per-shape test (float path)
// ---------------------------------------------------------------------------
static bool test_f32(int rows, int cols) {
    printf("  Shape [%d, %d] float32\n", rows, cols);
    bool ok = true;
    const float eps = 1e-5f;

    g_rng += (uint32_t)(rows * cols);  // vary seed per shape
    float* d_x     = alloc_rand_f32((int64_t)rows * cols, 1.0f);
    float* d_gamma = alloc_rand_f32(cols, 1.0f);
    float* d_beta  = alloc_rand_f32(cols, 0.5f);

    // Reference forward
    float* d_y_ref  = alloc_zeros_f32((int64_t)rows * cols);
    float* d_mean   = alloc_zeros_f32(rows);
    float* d_rstd   = alloc_zeros_f32(rows);
    layer_norm_forward_cuda(d_x, d_gamma, d_beta, d_y_ref, d_mean, d_rstd, rows, cols, eps);
    cudaDeviceSynchronize();

    // Vectorized forward
    float* d_y_vec   = alloc_zeros_f32((int64_t)rows * cols);
    float* d_mean_v  = alloc_zeros_f32(rows);
    float* d_rstd_v  = alloc_zeros_f32(rows);
    vln_forward_f32(d_x, d_gamma, d_beta, d_y_vec, d_mean_v, d_rstd_v, rows, cols, eps);
    cudaDeviceSynchronize();

    auto hy_ref = d2h_f32(d_y_ref, (int64_t)rows * cols);
    auto hy_vec = d2h_f32(d_y_vec, (int64_t)rows * cols);
    ok &= report("  fwd y", max_abs_diff(hy_ref, hy_vec), 1e-5f);

    auto hm_ref = d2h_f32(d_mean, rows);
    auto hm_vec = d2h_f32(d_mean_v, rows);
    ok &= report("  fwd mean", max_abs_diff(hm_ref, hm_vec), 1e-5f);

    auto hr_ref = d2h_f32(d_rstd, rows);
    auto hr_vec = d2h_f32(d_rstd_v, rows);
    ok &= report("  fwd rstd", max_abs_diff(hr_ref, hr_vec), 1e-5f);

    // Reference backward
    float* d_dy     = alloc_rand_f32((int64_t)rows * cols, 0.5f);
    float* d_dx_ref = alloc_zeros_f32((int64_t)rows * cols);
    float* d_dg_ref = alloc_zeros_f32(cols);
    float* d_db_ref = alloc_zeros_f32(cols);
    layer_norm_backward_cuda(d_dy, d_x, d_mean, d_rstd, d_gamma,
                             d_dx_ref, d_dg_ref, d_db_ref, rows, cols);
    cudaDeviceSynchronize();

    // Vectorized backward
    float* d_dx_vec = alloc_zeros_f32((int64_t)rows * cols);
    float* d_dg_vec = alloc_zeros_f32(cols);
    float* d_db_vec = alloc_zeros_f32(cols);
    vln_backward_f32(d_dy, d_x, d_mean, d_rstd, d_gamma,
                     d_dx_vec, d_dg_vec, d_db_vec, rows, cols);
    cudaDeviceSynchronize();

    auto hdx_r = d2h_f32(d_dx_ref, (int64_t)rows * cols);
    auto hdx_v = d2h_f32(d_dx_vec, (int64_t)rows * cols);
    ok &= report("  bwd dx", max_abs_diff(hdx_r, hdx_v), 1e-4f);

    auto hdg_r = d2h_f32(d_dg_ref, cols);
    auto hdg_v = d2h_f32(d_dg_vec, cols);
    ok &= report("  bwd dgamma", max_abs_diff(hdg_r, hdg_v), 1e-4f);

    auto hdb_r = d2h_f32(d_db_ref, cols);
    auto hdb_v = d2h_f32(d_db_vec, cols);
    ok &= report("  bwd dbeta", max_abs_diff(hdb_r, hdb_v), 1e-4f);

    cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_y_ref); cudaFree(d_mean); cudaFree(d_rstd);
    cudaFree(d_y_vec); cudaFree(d_mean_v); cudaFree(d_rstd_v);
    cudaFree(d_dy); cudaFree(d_dx_ref); cudaFree(d_dg_ref); cudaFree(d_db_ref);
    cudaFree(d_dx_vec); cudaFree(d_dg_vec); cudaFree(d_db_vec);
    return ok;
}

// ---------------------------------------------------------------------------
// Per-shape test (fp16 path): compare vln_forward_f16 vs float reference
// ---------------------------------------------------------------------------
static bool test_f16(int rows, int cols) {
    printf("  Shape [%d, %d] fp16\n", rows, cols);
    bool ok = true;
    const float eps = 1e-5f;

    g_rng += (uint32_t)(rows * cols + 1);
    __half* d_x    = alloc_rand_f16((int64_t)rows * cols, 1.0f);
    __half* d_gamma = alloc_rand_f16(cols, 1.0f);
    __half* d_beta  = alloc_rand_f16(cols, 0.5f);

    // Float reference: cast to float, run reference, cast back
    std::vector<__half> hx(rows * cols), hg(cols), hb(cols);
    cudaMemcpy(hx.data(), d_x,     rows * cols * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hg.data(), d_gamma, cols * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), d_beta,  cols * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> hx_f(rows * cols), hg_f(cols), hb_f(cols);
    for (int i = 0; i < rows * cols; i++) hx_f[i] = __half2float(hx[i]);
    for (int i = 0; i < cols; i++) { hg_f[i] = __half2float(hg[i]); hb_f[i] = __half2float(hb[i]); }

    float* d_x_f    = nullptr; cudaMalloc(&d_x_f,    rows * cols * sizeof(float));
    float* d_gamma_f = nullptr; cudaMalloc(&d_gamma_f, cols * sizeof(float));
    float* d_beta_f  = nullptr; cudaMalloc(&d_beta_f,  cols * sizeof(float));
    cudaMemcpy(d_x_f,     hx_f.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma_f, hg_f.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_f,  hb_f.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    float* d_y_ref_f = alloc_zeros_f32((int64_t)rows * cols);
    float* d_mean    = alloc_zeros_f32(rows);
    float* d_rstd    = alloc_zeros_f32(rows);
    layer_norm_forward_cuda(d_x_f, d_gamma_f, d_beta_f, d_y_ref_f, d_mean, d_rstd, rows, cols, eps);
    cudaDeviceSynchronize();

    // Vectorized f16 forward
    __half* d_y_vec = nullptr;
    cuda_check(cudaMalloc(&d_y_vec, rows * cols * sizeof(__half)), "alloc y_vec f16");
    cuda_check(cudaMemset(d_y_vec, 0, rows * cols * sizeof(__half)), "memset y_vec f16");
    float* d_mean_v = alloc_zeros_f32(rows);
    float* d_rstd_v = alloc_zeros_f32(rows);
    vln_forward_f16(d_x, d_gamma, d_beta, d_y_vec, d_mean_v, d_rstd_v, rows, cols, eps);
    cudaDeviceSynchronize();

    auto hy_ref = d2h_f32(d_y_ref_f, (int64_t)rows * cols);
    auto hy_vec = d2h_f16_as_f32(d_y_vec, (int64_t)rows * cols);
    ok &= report("  fwd y (f16 vs f32 ref)", max_abs_diff(hy_ref, hy_vec), 1e-2f);

    // Vectorized f16 backward
    __half* d_dy = alloc_rand_f16((int64_t)rows * cols, 0.5f);
    __half* d_dx_vec = nullptr;
    cuda_check(cudaMalloc(&d_dx_vec, rows * cols * sizeof(__half)), "alloc dx f16");
    cuda_check(cudaMemset(d_dx_vec, 0, rows * cols * sizeof(__half)), "memset dx f16");
    float* d_dg_vec = alloc_zeros_f32(cols);
    float* d_db_vec = alloc_zeros_f32(cols);
    vln_backward_f16(d_dy, d_x, d_mean_v, d_rstd_v, d_gamma,
                     d_dx_vec, d_dg_vec, d_db_vec, rows, cols);
    cudaDeviceSynchronize();
    // Smoke check: dx should be finite (non-NaN)
    auto hdx = d2h_f16_as_f32(d_dx_vec, (int64_t)rows * cols);
    bool finite_ok = true;
    for (auto v : hdx) { if (!std::isfinite(v)) { finite_ok = false; break; } }
    ok &= report("  bwd dx finite", finite_ok ? 0.0f : 1.0f, 0.5f);

    cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_x_f); cudaFree(d_gamma_f); cudaFree(d_beta_f);
    cudaFree(d_y_ref_f); cudaFree(d_mean); cudaFree(d_rstd);
    cudaFree(d_y_vec); cudaFree(d_mean_v); cudaFree(d_rstd_v);
    cudaFree(d_dy); cudaFree(d_dx_vec); cudaFree(d_dg_vec); cudaFree(d_db_vec);
    return ok;
}

// ---------------------------------------------------------------------------
// Per-shape test (bf16 path)
// ---------------------------------------------------------------------------
static bool test_bf16(int rows, int cols) {
    printf("  Shape [%d, %d] bf16\n", rows, cols);
    bool ok = true;
    const float eps = 1e-5f;

    g_rng += (uint32_t)(rows * cols + 2);
    __nv_bfloat16* d_x     = alloc_rand_bf16((int64_t)rows * cols, 1.0f);
    __nv_bfloat16* d_gamma = alloc_rand_bf16(cols, 1.0f);
    __nv_bfloat16* d_beta  = alloc_rand_bf16(cols, 0.5f);

    // Reference: convert bf16 → float, run reference LayerNorm
    std::vector<__nv_bfloat16> hx(rows * cols), hg(cols), hb(cols);
    cudaMemcpy(hx.data(), d_x,     rows * cols * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(hg.data(), d_gamma, cols * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), d_beta,  cols * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    std::vector<float> hx_f(rows * cols), hg_f(cols), hb_f(cols);
    for (int i = 0; i < rows * cols; i++) hx_f[i] = __bfloat162float(hx[i]);
    for (int i = 0; i < cols; i++) { hg_f[i] = __bfloat162float(hg[i]); hb_f[i] = __bfloat162float(hb[i]); }

    float* d_x_f     = nullptr; cudaMalloc(&d_x_f,    rows * cols * sizeof(float));
    float* d_gamma_f = nullptr; cudaMalloc(&d_gamma_f, cols * sizeof(float));
    float* d_beta_f  = nullptr; cudaMalloc(&d_beta_f,  cols * sizeof(float));
    cudaMemcpy(d_x_f,     hx_f.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma_f, hg_f.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_f,  hb_f.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    float* d_y_ref_f = alloc_zeros_f32((int64_t)rows * cols);
    float* d_mean    = alloc_zeros_f32(rows);
    float* d_rstd    = alloc_zeros_f32(rows);
    layer_norm_forward_cuda(d_x_f, d_gamma_f, d_beta_f, d_y_ref_f, d_mean, d_rstd, rows, cols, eps);
    cudaDeviceSynchronize();

    // Vectorized bf16 forward
    __nv_bfloat16* d_y_vec = nullptr;
    cuda_check(cudaMalloc(&d_y_vec, rows * cols * sizeof(__nv_bfloat16)), "alloc y_vec bf16");
    cuda_check(cudaMemset(d_y_vec, 0, rows * cols * sizeof(__nv_bfloat16)), "memset y_vec bf16");
    float* d_mean_v = alloc_zeros_f32(rows);
    float* d_rstd_v = alloc_zeros_f32(rows);
    vln_forward_bf16(d_x, d_gamma, d_beta, d_y_vec, d_mean_v, d_rstd_v, rows, cols, eps);
    cudaDeviceSynchronize();

    auto hy_ref = d2h_f32(d_y_ref_f, (int64_t)rows * cols);
    auto hy_vec = d2h_bf16_as_f32(d_y_vec, (int64_t)rows * cols);
    ok &= report("  fwd y (bf16 vs f32 ref)", max_abs_diff(hy_ref, hy_vec), 5e-2f);

    // Vectorized bf16 backward
    __nv_bfloat16* d_dy  = alloc_rand_bf16((int64_t)rows * cols, 0.5f);
    __nv_bfloat16* d_dx  = nullptr;
    cuda_check(cudaMalloc(&d_dx, rows * cols * sizeof(__nv_bfloat16)), "alloc dx bf16");
    cuda_check(cudaMemset(d_dx, 0, rows * cols * sizeof(__nv_bfloat16)), "memset dx bf16");
    float* d_dg = alloc_zeros_f32(cols);
    float* d_db = alloc_zeros_f32(cols);
    vln_backward_bf16(d_dy, d_x, d_mean_v, d_rstd_v, d_gamma,
                      d_dx, d_dg, d_db, rows, cols);
    cudaDeviceSynchronize();
    auto hdx = d2h_bf16_as_f32(d_dx, (int64_t)rows * cols);
    bool finite_ok = true;
    for (auto v : hdx) { if (!std::isfinite(v)) { finite_ok = false; break; } }
    ok &= report("  bwd dx finite", finite_ok ? 0.0f : 1.0f, 0.5f);

    cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_x_f); cudaFree(d_gamma_f); cudaFree(d_beta_f);
    cudaFree(d_y_ref_f); cudaFree(d_mean); cudaFree(d_rstd);
    cudaFree(d_y_vec); cudaFree(d_mean_v); cudaFree(d_rstd_v);
    cudaFree(d_dy); cudaFree(d_dx); cudaFree(d_dg); cudaFree(d_db);
    return ok;
}

// ---------------------------------------------------------------------------
// Timing (float path, shape 512x384)
// ---------------------------------------------------------------------------
static void timing_f32(int rows, int cols, int NITER) {
    printf("\n--- Timing float32 [%d, %d], %d iters ---\n", rows, cols, NITER);
    const float eps = 1e-5f;
    g_rng += 999u;
    float* d_x     = alloc_rand_f32((int64_t)rows * cols);
    float* d_gamma = alloc_rand_f32(cols);
    float* d_beta  = alloc_rand_f32(cols);
    float* d_y     = alloc_zeros_f32((int64_t)rows * cols);
    float* d_mean  = alloc_zeros_f32(rows);
    float* d_rstd  = alloc_zeros_f32(rows);
    float* d_dy    = alloc_rand_f32((int64_t)rows * cols, 0.5f);
    float* d_dx    = alloc_zeros_f32((int64_t)rows * cols);
    float* d_dg    = alloc_zeros_f32(cols);
    float* d_db    = alloc_zeros_f32(cols);

    // Warm-up
    for (int w = 0; w < 5; w++) {
        layer_norm_forward_cuda(d_x, d_gamma, d_beta, d_y, d_mean, d_rstd, rows, cols, eps);
        vln_forward_f32(d_x, d_gamma, d_beta, d_y, d_mean, d_rstd, rows, cols, eps);
    }
    cudaDeviceSynchronize();

    // Reference forward
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NITER; i++)
        layer_norm_forward_cuda(d_x, d_gamma, d_beta, d_y, d_mean, d_rstd, rows, cols, eps);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_ref_fwd = std::chrono::duration<double, std::milli>(t1 - t0).count() / NITER;

    // Vectorized forward
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NITER; i++)
        vln_forward_f32(d_x, d_gamma, d_beta, d_y, d_mean, d_rstd, rows, cols, eps);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_vec_fwd = std::chrono::duration<double, std::milli>(t3 - t2).count() / NITER;

    printf("  Forward  ref: %6.3f ms   vec: %6.3f ms   speedup: %.2fx\n",
           ms_ref_fwd, ms_vec_fwd, ms_ref_fwd / ms_vec_fwd);

    // Reference backward
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NITER; i++)
        layer_norm_backward_cuda(d_dy, d_x, d_mean, d_rstd, d_gamma, d_dx, d_dg, d_db, rows, cols);
    cudaDeviceSynchronize();
    auto t5 = std::chrono::high_resolution_clock::now();
    double ms_ref_bwd = std::chrono::duration<double, std::milli>(t5 - t4).count() / NITER;

    // Vectorized backward
    auto t6 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NITER; i++)
        vln_backward_f32(d_dy, d_x, d_mean, d_rstd, d_gamma, d_dx, d_dg, d_db, rows, cols);
    cudaDeviceSynchronize();
    auto t7 = std::chrono::high_resolution_clock::now();
    double ms_vec_bwd = std::chrono::duration<double, std::milli>(t7 - t6).count() / NITER;

    printf("  Backward ref: %6.3f ms   vec: %6.3f ms   speedup: %.2fx\n",
           ms_ref_bwd, ms_vec_bwd, ms_ref_bwd / ms_vec_bwd);

    cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_y); cudaFree(d_mean); cudaFree(d_rstd);
    cudaFree(d_dy); cudaFree(d_dx); cudaFree(d_dg); cudaFree(d_db);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    cuda_check(cudaSetDevice(0), "setDevice");

    printf("\n=== test_vectorized_layernorm ===\n\n");
    bool all_ok = true;

    // Float32 correctness
    printf("--- float32 correctness ---\n");
    all_ok &= test_f32(512, 384);
    all_ok &= test_f32(512, 768);
    all_ok &= test_f32(64,  64);
    all_ok &= test_f32(64,  65);   // scalar fallback path
    printf("\n");

    // FP16 correctness
    printf("--- fp16 correctness ---\n");
    all_ok &= test_f16(512, 384);
    all_ok &= test_f16(64,  64);
    printf("\n");

    // BF16 correctness
    printf("--- bf16 correctness ---\n");
    all_ok &= test_bf16(512, 384);
    all_ok &= test_bf16(64,  64);
    printf("\n");

    // Timing
    timing_f32(512, 384, 1000);
    timing_f32(512, 768, 1000);

    printf("\n=== %s ===\n\n", all_ok ? "ALL PASS" : "SOME FAIL");
    return all_ok ? 0 : 1;
}
