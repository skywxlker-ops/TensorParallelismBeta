// =============================================================================
// bench_adamw_isolated.cpp
//
// Standalone micro-benchmark for multi_tensor_adam_cuda.
// NO model, NO training loop, NO MPI, NO NCCL.
//
// Allocates param/grad/m/v tensors matching the GPT-2 (384 embd, 3 layers)
// parameter layout (~44M elements) and times pure AdamW kernel calls.
//
// Also monitors GPU clocks during the benchmark to detect downclocking.
//
// Build:  make bench_adamw_isolated
// Run:    ./bench_adamw_isolated_exec
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#include <cuda_runtime.h>

#include "ops/helpers/MultiTensorKernels.h"
#include "ops/helpers/KernelDispatch.h"

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

// Print GPU clock and power info
static void print_gpu_state(const char* label) {
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, 0), "getDeviceProperties");
    
    // Use nvml-like queries through cudaDeviceGetAttribute
    int clockRate, memClockRate;
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);           // in kHz
    cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, 0);   // in kHz
    
    printf("[GPU STATE: %s]\n", label);
    printf("  Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("  SMs: %d\n", prop.multiProcessorCount);
    printf("  Max Clock: %d MHz, Memory Clock: %d MHz\n", clockRate/1000, memClockRate/1000);
    printf("  ArchFamily: %d\n", (int)get_arch(0));
}

// ---------------------------------------------------------------------------
// GPT-2 124M-like parameter shapes (n_embd=384, n_layers=3, vocab=50304)
// This matches the actual parameter layout from the training script.
// ---------------------------------------------------------------------------

struct ParamSpec {
    const char* name;
    int64_t numel;
};

static std::vector<ParamSpec> make_gpt2_param_specs() {
    const int n_embd = 384;
    const int n_heads = 6;
    const int n_layers = 3;
    const int vocab_size = 50304;
    const int ctx_len = 1024;
    
    std::vector<ParamSpec> specs;
    
    // wte: [vocab_size, n_embd]
    specs.push_back({"wte.weight", (int64_t)vocab_size * n_embd});
    // wpe: [ctx_len, n_embd]
    specs.push_back({"wpe.weight", (int64_t)ctx_len * n_embd});
    
    for (int L = 0; L < n_layers; L++) {
        // Attention block
        // ln weight + bias
        specs.push_back({"attn.ln.weight", n_embd});
        specs.push_back({"attn.ln.bias", n_embd});
        // qkv weight + bias  [n_embd, 3*n_embd]
        specs.push_back({"attn.qkv.weight", (int64_t)n_embd * 3 * n_embd});
        specs.push_back({"attn.qkv.bias", 3 * n_embd});
        // proj weight + bias [n_embd, n_embd]
        specs.push_back({"attn.proj.weight", (int64_t)n_embd * n_embd});
        specs.push_back({"attn.proj.bias", n_embd});
        
        // MLP block
        // ln weight + bias
        specs.push_back({"mlp.ln.weight", n_embd});
        specs.push_back({"mlp.ln.bias", n_embd});
        // fc_up weight + bias [n_embd, 4*n_embd]
        specs.push_back({"mlp.fc_up.weight", (int64_t)n_embd * 4 * n_embd});
        specs.push_back({"mlp.fc_up.bias", 4 * n_embd});
        // fc_down weight + bias [4*n_embd, n_embd]
        specs.push_back({"mlp.fc_down.weight", (int64_t)4 * n_embd * n_embd});
        specs.push_back({"mlp.fc_down.bias", n_embd});
    }
    
    // ln_f weight + bias
    specs.push_back({"ln_f.weight", n_embd});
    specs.push_back({"ln_f.bias", n_embd});
    // lm_head weight (no weight tying) [n_embd, vocab_size]
    specs.push_back({"lm_head.weight", (int64_t)n_embd * vocab_size});
    
    return specs;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    cuda_check(cudaSetDevice(0), "setDevice");
    
    printf("\n");
    printf("==========================================================\n");
    printf("  AdamW Micro-Benchmark (Isolated — No Model/MPI/NCCL)\n");
    printf("==========================================================\n\n");
    
    // Print GPU state before anything
    print_gpu_state("BEFORE BENCHMARK");
    printf("\n");
    
    // Build parameter specs matching GPT-2
    auto specs = make_gpt2_param_specs();
    int N = (int)specs.size();
    
    int64_t total_numel = 0;
    for (auto& s : specs) total_numel += s.numel;
    
    printf("Parameter layout: %d tensors, %lld total elements (%.2f M)\n",
           N, (long long)total_numel, total_numel / 1e6);
    printf("\n");
    
    // Allocate GPU arrays
    uint32_t rng = 0xDEADBEEFu;
    
    std::vector<float*> d_params(N), d_grads(N), d_ms(N), d_vs(N);
    for (int i = 0; i < N; i++) {
        d_params[i] = make_device_array(specs[i].numel, rng, 0.02f);
        d_grads[i]  = make_device_array(specs[i].numel, rng, 0.01f);
        d_ms[i]     = make_device_array(specs[i].numel, rng, 0.001f);
        d_vs[i]     = make_device_array(specs[i].numel, rng, 0.0001f);
    }
    
    // Build TensorInfo vectors
    std::vector<TensorInfo> ti_params(N), ti_grads(N), ti_ms(N), ti_vs(N);
    for (int i = 0; i < N; i++) {
        ti_params[i] = {d_params[i], specs[i].numel};
        ti_grads[i]  = {d_grads[i],  specs[i].numel};
        ti_ms[i]     = {d_ms[i],     specs[i].numel};
        ti_vs[i]     = {d_vs[i],     specs[i].numel};
    }
    
    // Adam hyperparams (matching the training script)
    const float lr   = 6e-4f;
    const float b1   = 0.9f;
    const float b2   = 0.95f;
    const float eps  = 1e-8f;
    const float wd   = 0.1f;
    const float bc1  = 1.0f - powf(b1, 1.0f);  // step=1
    const float bc2  = 1.0f - powf(b2, 1.0f);
    
    printf("Adam config: lr=%.1e, beta1=%.2f, beta2=%.2f, eps=%.1e, wd=%.2f\n",
           lr, b1, b2, eps, wd);
    printf("bias_correction1=%.6f, bias_correction2=%.6f\n\n", bc1, bc2);
    
    // ===================================================================
    // PHASE 1: Warm-up (5 calls to let GPU boost clocks)
    // ===================================================================
    printf("--- Phase 1: Warm-up (5 calls) ---\n");
    for (int w = 0; w < 5; w++) {
        multi_tensor_adam_cuda(ti_params, ti_grads, ti_ms, ti_vs,
                               lr, b1, b2, eps, wd, bc1, bc2, /*is_adamw=*/true);
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");
    printf("  Done.\n\n");
    
    // ===================================================================
    // PHASE 2: Individual step timing (CUDA events, 20 calls)
    // ===================================================================
    printf("--- Phase 2: Individual step timing (20 calls, CUDA events) ---\n");
    
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "event create");
    cuda_check(cudaEventCreate(&stop), "event create");
    
    for (int i = 0; i < 20; i++) {
        cuda_check(cudaEventRecord(start, 0), "event record");
        multi_tensor_adam_cuda(ti_params, ti_grads, ti_ms, ti_vs,
                               lr, b1, b2, eps, wd, bc1, bc2, true);
        cuda_check(cudaEventRecord(stop, 0), "event record");
        cuda_check(cudaEventSynchronize(stop), "event sync");
        
        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        printf("  call %2d: %.3f ms\n", i, ms);
    }
    printf("\n");
    
    // ===================================================================
    // PHASE 3: Steady-state batch timing (100 calls, wall clock)
    // ===================================================================
    printf("--- Phase 3: Steady-state batch (100 calls, wall clock) ---\n");
    
    cuda_check(cudaDeviceSynchronize(), "pre-batch sync");
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        multi_tensor_adam_cuda(ti_params, ti_grads, ti_ms, ti_vs,
                               lr, b1, b2, eps, wd, bc1, bc2, true);
    }
    cuda_check(cudaDeviceSynchronize(), "post-batch sync");
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_avg = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100.0;
    printf("  Average: %.3f ms/call\n\n", ms_avg);
    
    // ===================================================================
    // PHASE 4: Check GPU state after benchmark
    // ===================================================================
    print_gpu_state("AFTER BENCHMARK");
    printf("\n");
    
    // ===================================================================
    // Summary
    // ===================================================================
    printf("==========================================================\n");
    printf("  SUMMARY\n");
    printf("  Total params:    %lld (%.2f M)\n", (long long)total_numel, total_numel/1e6);
    printf("  Num tensors:     %d\n", N);
    printf("  Avg time/call:   %.3f ms  (batch of 100)\n", ms_avg);
    printf("  Expected:        ~3.8 ms (friend's machine)\n");
    printf("  Bandwidth:       %.2f GB/s (4 arrays R+W = 8 * numel * 4B)\n",
           (8.0 * total_numel * 4.0) / (ms_avg * 1e-3) / 1e9);
    printf("==========================================================\n\n");
    
    // Cleanup
    cuda_check(cudaEventDestroy(start), "event destroy");
    cuda_check(cudaEventDestroy(stop), "event destroy");
    for (int i = 0; i < N; i++) {
        cudaFree(d_params[i]); cudaFree(d_grads[i]);
        cudaFree(d_ms[i]);     cudaFree(d_vs[i]);
    }
    
    return 0;
}
