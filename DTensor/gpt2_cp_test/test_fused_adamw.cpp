// =============================================================================
// test_fused_adamw.cpp
//
// Correctness + timing test for FusedAdamWKernel.
//
// RED: This file references fused_adamw_with_unscale_cuda() which does not
//      exist until FusedAdamWKernel.cu is compiled. Build will fail until
//      the implementation is added.
//
// GREEN: Once FusedAdamWKernel.cu is built, all PASS checks should hold.
//
// Correctness:
//   With inv_scale=1.0 and clip_coeff=1.0, fused_adamw_with_unscale_cuda
//   must produce param/m/v results identical to multi_tensor_adam_cuda
//   (max abs diff < 1e-5 for each).
//
//   With inv_scale=0.5, each gradient is halved before the Adam update.
//   The test verifies params differ from inv_scale=1.0 by a predictable amount.
//
// Timing:
//   N_TENSORS=50 tensors of 4096 elements each (total ~200k params).
//   Reference:  multi_tensor_adam_cuda (no unscale)
//   Fused:      fused_adamw_with_unscale_cuda (inv_scale=1.0)
//   Expected:   fused >= reference speed (same kernel, slightly more work).
//
// Build:  make test_fused_adamw
// Run:    ./test_fused_adamw_exec
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>

#include <cuda_runtime.h>

#include "ops/helpers/MultiTensorKernels.h"
#include "dnn/FusedAdamWKernel.h"

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

// Simple LCG for reproducible pseudo-random floats in [-0.5, 0.5]
static float lcg_randf(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return (static_cast<float>(state >> 8) / static_cast<float>(1 << 24)) - 0.5f;
}

// Allocate and fill a device float array from host random data
static float* make_device_array(int64_t n, uint32_t& rng_state, float scale = 1.0f) {
    std::vector<float> h(n);
    for (auto& v : h) v = lcg_randf(rng_state) * scale;
    float* d = nullptr;
    cuda_check(cudaMalloc(&d, n * sizeof(float)), "malloc");
    cuda_check(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice), "memcpy H2D");
    return d;
}

// Copy device array to host vector
static std::vector<float> to_host(const float* d, int64_t n) {
    std::vector<float> h(n);
    cuda_check(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost), "memcpy D2H");
    return h;
}

// Deep-copy a device array
static float* clone_device(const float* src, int64_t n) {
    float* dst = nullptr;
    cuda_check(cudaMalloc(&dst, n * sizeof(float)), "malloc clone");
    cuda_check(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice), "memcpy D2D");
    return dst;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
        mx = std::max(mx, std::abs(a[i] - b[i]));
    return mx;
}

static bool check(const char* label, float diff, float tol) {
    bool ok = diff < tol;
    printf("  [%s] max_abs_diff = %.2e  %s\n", label, diff, ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test configuration
// ---------------------------------------------------------------------------

static const int N_TENSORS  = 50;
static const int ELEM       = 4096;
static const int TIMING_REF = 200;
static const int TIMING_FUS = 200;

// Adam hyperparams
static const float LR    = 1e-3f;
static const float B1    = 0.9f;
static const float B2    = 0.999f;
static const float EPS   = 1e-8f;
static const float WD    = 0.1f;
static const float BC1   = 1.0f - powf(B1, 1.0f);  // step=1
static const float BC2   = 1.0f - powf(B2, 1.0f);

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    cuda_check(cudaSetDevice(0), "setDevice");

    printf("\n=== test_fused_adamw ===\n\n");

    // -------------------------------------------------------------------
    // Allocate shared initial state for all tensors
    // -------------------------------------------------------------------
    uint32_t rng = 0xDEADBEEFu;

    // Each tensor: param, grad, m, v
    std::vector<float*> d_param0(N_TENSORS);
    std::vector<float*> d_grad(N_TENSORS);
    std::vector<float*> d_m0(N_TENSORS);
    std::vector<float*> d_v0(N_TENSORS);

    for (int i = 0; i < N_TENSORS; i++) {
        d_param0[i] = make_device_array(ELEM, rng, 0.5f);
        d_grad[i]   = make_device_array(ELEM, rng, 0.1f);
        d_m0[i]     = make_device_array(ELEM, rng, 0.01f);
        d_v0[i]     = make_device_array(ELEM, rng, 0.001f);
    }

    // -------------------------------------------------------------------
    // Reference: multi_tensor_adam_cuda (no unscale)
    // -------------------------------------------------------------------
    std::vector<float*> d_param_ref(N_TENSORS);
    std::vector<float*> d_m_ref(N_TENSORS);
    std::vector<float*> d_v_ref(N_TENSORS);
    for (int i = 0; i < N_TENSORS; i++) {
        d_param_ref[i] = clone_device(d_param0[i], ELEM);
        d_m_ref[i]     = clone_device(d_m0[i],    ELEM);
        d_v_ref[i]     = clone_device(d_v0[i],    ELEM);
    }

    std::vector<TensorInfo> ref_params(N_TENSORS), ref_grads(N_TENSORS);
    std::vector<TensorInfo> ref_ms(N_TENSORS),    ref_vs(N_TENSORS);
    for (int i = 0; i < N_TENSORS; i++) {
        ref_params[i] = {d_param_ref[i], ELEM};
        ref_grads[i]  = {d_grad[i],      ELEM};
        ref_ms[i]     = {d_m_ref[i],     ELEM};
        ref_vs[i]     = {d_v_ref[i],     ELEM};
    }

    multi_tensor_adam_cuda(ref_params, ref_grads, ref_ms, ref_vs,
                           LR, B1, B2, EPS, WD, BC1, BC2, /*is_adamw=*/true);
    cuda_check(cudaDeviceSynchronize(), "ref sync");

    // -------------------------------------------------------------------
    // Fused: inv_scale=1.0, clip_coeff=1.0  (must match reference)
    // -------------------------------------------------------------------
    std::vector<float*> d_param_fus(N_TENSORS);
    std::vector<float*> d_m_fus(N_TENSORS);
    std::vector<float*> d_v_fus(N_TENSORS);
    for (int i = 0; i < N_TENSORS; i++) {
        d_param_fus[i] = clone_device(d_param0[i], ELEM);
        d_m_fus[i]     = clone_device(d_m0[i],    ELEM);
        d_v_fus[i]     = clone_device(d_v0[i],    ELEM);
    }

    std::vector<TensorInfo> fus_params(N_TENSORS), fus_grads(N_TENSORS);
    std::vector<TensorInfo> fus_ms(N_TENSORS),    fus_vs(N_TENSORS);
    for (int i = 0; i < N_TENSORS; i++) {
        fus_params[i] = {d_param_fus[i], ELEM};
        fus_grads[i]  = {d_grad[i],      ELEM};
        fus_ms[i]     = {d_m_fus[i],     ELEM};
        fus_vs[i]     = {d_v_fus[i],     ELEM};
    }

    fused_adamw_with_unscale_cuda(fus_params, fus_grads, fus_ms, fus_vs,
                                  LR, B1, B2, EPS, WD, BC1, BC2,
                                  /*inv_scale=*/1.0f, /*clip_coeff=*/1.0f);
    cuda_check(cudaDeviceSynchronize(), "fused sync");

    // -------------------------------------------------------------------
    // Correctness check: inv_scale=1.0 must match reference exactly
    // -------------------------------------------------------------------
    printf("--- Correctness (inv_scale=1.0, clip=1.0 vs reference) ---\n");
    bool all_ok = true;
    float max_param = 0.0f, max_m = 0.0f, max_v = 0.0f;
    for (int i = 0; i < N_TENSORS; i++) {
        auto hp = to_host(d_param_ref[i], ELEM);
        auto hf = to_host(d_param_fus[i], ELEM);
        max_param = std::max(max_param, max_abs_diff(hp, hf));

        auto hm_r = to_host(d_m_ref[i], ELEM);
        auto hm_f = to_host(d_m_fus[i], ELEM);
        max_m = std::max(max_m, max_abs_diff(hm_r, hm_f));

        auto hv_r = to_host(d_v_ref[i], ELEM);
        auto hv_f = to_host(d_v_fus[i], ELEM);
        max_v = std::max(max_v, max_abs_diff(hv_r, hv_f));
    }
    all_ok &= check("param  (inv_scale=1.0)", max_param, 1e-5f);
    all_ok &= check("m      (inv_scale=1.0)", max_m,     1e-5f);
    all_ok &= check("v      (inv_scale=1.0)", max_v,     1e-5f);
    printf("\n");

    // -------------------------------------------------------------------
    // Correctness check: inv_scale=0.5 → grads halved → params must differ
    // from inv_scale=1.0 by a non-trivial amount
    // -------------------------------------------------------------------
    printf("--- Correctness (inv_scale=0.5 behaves differently from 1.0) ---\n");
    {
        std::vector<float*> d_ph(N_TENSORS), d_mh(N_TENSORS), d_vh(N_TENSORS);
        for (int i = 0; i < N_TENSORS; i++) {
            d_ph[i] = clone_device(d_param0[i], ELEM);
            d_mh[i] = clone_device(d_m0[i],    ELEM);
            d_vh[i] = clone_device(d_v0[i],    ELEM);
        }
        std::vector<TensorInfo> ph(N_TENSORS), gh(N_TENSORS), mh(N_TENSORS), vh(N_TENSORS);
        for (int i = 0; i < N_TENSORS; i++) {
            ph[i] = {d_ph[i], ELEM};
            gh[i] = {d_grad[i], ELEM};
            mh[i] = {d_mh[i], ELEM};
            vh[i] = {d_vh[i], ELEM};
        }
        fused_adamw_with_unscale_cuda(ph, gh, mh, vh,
                                      LR, B1, B2, EPS, WD, BC1, BC2,
                                      /*inv_scale=*/0.5f, /*clip_coeff=*/1.0f);
        cuda_check(cudaDeviceSynchronize(), "half sync");

        float max_diff = 0.0f;
        for (int i = 0; i < N_TENSORS; i++) {
            auto href = to_host(d_param_fus[i], ELEM);  // inv_scale=1.0 result
            auto hhalf = to_host(d_ph[i], ELEM);
            max_diff = std::max(max_diff, max_abs_diff(href, hhalf));
        }
        bool ok = max_diff > 1e-7f;
        printf("  [param diff (0.5 vs 1.0)] max_abs_diff = %.2e  %s\n",
               max_diff, ok ? "PASS (params differ as expected)" : "FAIL (params identical — inv_scale not applied)");
        all_ok &= ok;

        for (int i = 0; i < N_TENSORS; i++) {
            cudaFree(d_ph[i]); cudaFree(d_mh[i]); cudaFree(d_vh[i]);
        }
    }
    printf("\n");

    // -------------------------------------------------------------------
    // Timing
    // -------------------------------------------------------------------
    printf("--- Timing (%d tensors x %d elements, %d iterations) ---\n",
           N_TENSORS, ELEM, TIMING_REF);

    // Warm-up
    for (int w = 0; w < 5; w++) {
        multi_tensor_adam_cuda(ref_params, ref_grads, ref_ms, ref_vs,
                               LR, B1, B2, EPS, WD, BC1, BC2, true);
        fused_adamw_with_unscale_cuda(fus_params, fus_grads, fus_ms, fus_vs,
                                      LR, B1, B2, EPS, WD, BC1, BC2, 1.0f, 1.0f);
    }
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < TIMING_REF; it++)
        multi_tensor_adam_cuda(ref_params, ref_grads, ref_ms, ref_vs,
                               LR, B1, B2, EPS, WD, BC1, BC2, true);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_ref = std::chrono::duration<double, std::milli>(t1 - t0).count() / TIMING_REF;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < TIMING_FUS; it++)
        fused_adamw_with_unscale_cuda(fus_params, fus_grads, fus_ms, fus_vs,
                                      LR, B1, B2, EPS, WD, BC1, BC2, 1.0f, 1.0f);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_fus = std::chrono::duration<double, std::milli>(t3 - t2).count() / TIMING_FUS;

    printf("  multi_tensor_adam_cuda            : %6.3f ms/step\n", ms_ref);
    printf("  fused_adamw_with_unscale_cuda     : %6.3f ms/step\n", ms_fus);
    printf("  Speedup (fused vs reference)      : %.2fx\n", ms_ref / ms_fus);
    printf("\n");

    // -------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------
    for (int i = 0; i < N_TENSORS; i++) {
        cudaFree(d_param0[i]); cudaFree(d_grad[i]);
        cudaFree(d_m0[i]);     cudaFree(d_v0[i]);
        cudaFree(d_param_ref[i]); cudaFree(d_m_ref[i]); cudaFree(d_v_ref[i]);
        cudaFree(d_param_fus[i]); cudaFree(d_m_fus[i]); cudaFree(d_v_fus[i]);
    }

    printf("=== %s ===\n\n", all_ok ? "ALL PASS" : "SOME FAIL");
    return all_ok ? 0 : 1;
}
