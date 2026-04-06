// =============================================================================
// CP Ring Attention Benchmark — Evaluation Harness
//
// Combined correctness + performance benchmark for the autoresearch tool.
//
// Output (stdout, machine-readable):
//   CORRECT=1|0          — did forward+backward match standard SDPA?
//   FWD_MS=<float>       — forward-only wall time
//   BWD_MS=<float>       — backward-only wall time (derived)
//   METRIC_MS=<float>    — THE SINGLE NUMBER (lower is better)
//
// The metric is:
//   if CORRECT:  METRIC_MS = fwd+bwd time
//   if !CORRECT: METRIC_MS = 99999.0   (penalty — tool must never pick this)
//
// Config: B=4 H=6 T_local=512 D=64 (T_full=1024, cp=2 GPUs, FP32)
//
// Build: make cp_ring_attn_bench
// Run:   mpirun -np 2 ./cp_ring_attn_bench_exec
// =============================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/BinaryOps.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/TensorOps.h"

#include "tensor/dtensor.h"
#include "tensor/device_mesh.h"
#include "process_group/ProcessGroupNCCL.h"
#include "gpt2_cp_test/context_parallel/ContextParallel.h"

using namespace OwnTensor;

// ---- helpers ----------------------------------------------------------------
static float max_abs_diff(const Tensor& a, const Tensor& b) {
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    int64_t n = a.numel();
    float d = 0.0f;
    for (int64_t i = 0; i < n; ++i) d = std::max(d, std::abs(pa[i] - pb[i]));
    return d;
}

// Standard (non-CP) attention forward — reference implementation
static Tensor standard_sdpa_forward(Tensor& q, Tensor& k, Tensor& v, float scale) {
    Shape s({{1}});
    TensorOptions so = TensorOptions().with_dtype(q.dtype()).with_device(q.device());
    Tensor st = Tensor::full(s, so, scale);
    Tensor qs = autograd::mul(q, st);
    Tensor kt = autograd::transpose(k, -2, -1);
    Tensor sc_ = autograd::matmul(qs, kt);
    float neg_inf = -std::numeric_limits<float>::infinity();
    Tensor masked = autograd::tril(sc_, 0, neg_inf);
    Tensor probs = autograd::softmax(masked);
    return autograd::matmul(probs, v);
}

// ---- main -------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (rank == 0) std::cerr << "ERROR: requires exactly 2 GPUs\n";
        MPI_Finalize();
        return 1;
    }

    cudaSetDevice(rank);

    std::vector<int> ranks_vec = {0, 1};
    DeviceMesh mesh({2}, ranks_vec);
    auto pg = mesh.get_process_group(0);
    DeviceIndex device(Device::CUDA, rank);

    // =========================================================================
    // Phase 1 — Correctness gate
    //
    // Small config (B=2 H=2 T=8 D=64) so the TC WMMA path is exercised
    // (D=64 is divisible by 16 → hits the unified Q-parallel kernel).
    // Compare forward output + backward dQ/dK/dV against standard SDPA.
    // =========================================================================
    const float GRAD_TOL = 1e-2f;   // allow TF32 rounding
    const float FWD_TOL  = 1e-3f;
    bool correct = true;

    {
        const int64_t Bc = 2, Hc = 2, Tc = 64, Dc = 64;
        const float sc = 1.0f / std::sqrt(static_cast<float>(Dc));

        Shape sh({{Bc, Hc, Tc, Dc}});
        TensorOptions opts = TensorOptions()
            .with_dtype(Dtype::Float32).with_device(device).with_req_grad(true);

        // Identical tensors on both ranks (same seed)
        Tensor q_full = Tensor::randn<float>(sh, opts, 400, 0.5f);
        Tensor k_full = Tensor::randn<float>(sh, opts, 401, 0.5f);
        Tensor v_full = Tensor::randn<float>(sh, opts, 402, 0.5f);

        // --- Standard reference (rank 0 only) ---
        Tensor std_out, std_dq, std_dk, std_dv;
        if (rank == 0) {
            Tensor qs = q_full.clone(); qs.set_requires_grad(true);
            Tensor ks = k_full.clone(); ks.set_requires_grad(true);
            Tensor vs = v_full.clone(); vs.set_requires_grad(true);
            std_out = standard_sdpa_forward(qs, ks, vs, sc);
            Tensor ones = Tensor::full(std_out.shape(),
                            std_out.opts().with_req_grad(false), 1.0f);
            std_out.backward(&ones);
            std_dq = qs.grad_view().to_cpu();
            std_dk = ks.grad_view().to_cpu();
            std_dv = vs.grad_view().to_cpu();
            std_out = std_out.to_cpu();
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // --- CP path (both ranks) ---
        {
            Tensor qc = q_full.clone(); qc.set_requires_grad(true);
            Tensor kc = k_full.clone(); kc.set_requires_grad(true);
            Tensor vc = v_full.clone(); vc.set_requires_grad(true);
            ContextParallel cp_c(mesh, pg, sc, true, RotatorType::AlltoAll, false);
            Tensor cp_out = cp_c.forward_cp(qc, kc, vc);
            Tensor ones = Tensor::full(cp_out.shape(),
                            cp_out.opts().with_req_grad(false), 1.0f);
            cp_out.backward(&ones);

            if (rank == 0) {
                Tensor cp_out_cpu = cp_out.to_cpu();
                Tensor cp_dq = qc.grad_view().to_cpu();
                Tensor cp_dk = kc.grad_view().to_cpu();
                Tensor cp_dv = vc.grad_view().to_cpu();

                float fwd_diff = max_abs_diff(std_out, cp_out_cpu);
                float dq_diff  = max_abs_diff(std_dq, cp_dq);
                float dk_diff  = max_abs_diff(std_dk, cp_dk);
                float dv_diff  = max_abs_diff(std_dv, cp_dv);

                bool fwd_ok = (fwd_diff < FWD_TOL);
                bool bwd_ok = (dq_diff < GRAD_TOL && dk_diff < GRAD_TOL && dv_diff < GRAD_TOL);
                correct = fwd_ok && bwd_ok;

                std::cerr << std::scientific;
                std::cerr << "[correctness] fwd max_diff=" << fwd_diff
                          << (fwd_ok ? " OK" : " FAIL") << "\n";
                std::cerr << "[correctness] dQ max_diff=" << dq_diff
                          << " dK max_diff=" << dk_diff
                          << " dV max_diff=" << dv_diff
                          << (bwd_ok ? " OK" : " FAIL") << "\n";
            }
        }
    }

    // Broadcast correctness flag to all ranks
    int correct_int = correct ? 1 : 0;
    MPI_Bcast(&correct_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    correct = (correct_int == 1);

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Phase 2 — Timing (full training-scale config)
    //
    // B=4 H=6 T_local=512 D=64  (T_full=1024, cp=2 GPUs)
    // =========================================================================
    const int64_t B = 4, H = 6, Tl = 512, D = 64;
    const float sc = 1.0f / std::sqrt(static_cast<float>(D));
    const int NWARM = 5, NITERS = 20;

    Shape qkv_shape({{B, H, Tl, D}});
    TensorOptions opts_rg = TensorOptions()
        .with_dtype(Dtype::Float32).with_device(device).with_req_grad(true);
    TensorOptions opts_ng = opts_rg.with_req_grad(false);

    // Warmup
    for (int w = 0; w < NWARM; ++w) {
        Tensor qw = Tensor::randn<float>(qkv_shape, opts_rg, 90 + w, 0.3f);
        Tensor kw = Tensor::randn<float>(qkv_shape, opts_rg, 91 + w, 0.3f);
        Tensor vw = Tensor::randn<float>(qkv_shape, opts_rg, 92 + w, 0.3f);
        ContextParallel cp_w(mesh, pg, sc, true, RotatorType::AlltoAll, false);
        Tensor out_w = cp_w.forward_cp(qw, kw, vw);
        Tensor ones_w = Tensor::ones(out_w.shape(), opts_ng);
        out_w.backward(&ones_w);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Timed: Forward only ---
    auto tf0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NITERS; ++i) {
        Tensor qi = Tensor::randn<float>(qkv_shape, opts_ng, 200 + i, 0.3f);
        Tensor ki = Tensor::randn<float>(qkv_shape, opts_ng, 201 + i, 0.3f);
        Tensor vi = Tensor::randn<float>(qkv_shape, opts_ng, 202 + i, 0.3f);
        ContextParallel cp_i(mesh, pg, sc, true, RotatorType::AlltoAll, false);
        cp_i.forward_cp(qi, ki, vi);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    auto tf1 = std::chrono::high_resolution_clock::now();
    const double ms_fwd = std::chrono::duration<double, std::milli>(tf1 - tf0).count() / NITERS;

    // --- Timed: Forward + Backward ---
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NITERS; ++i) {
        Tensor qi = Tensor::randn<float>(qkv_shape, opts_rg, 100 + i, 0.3f);
        Tensor ki = Tensor::randn<float>(qkv_shape, opts_rg, 101 + i, 0.3f);
        Tensor vi = Tensor::randn<float>(qkv_shape, opts_rg, 102 + i, 0.3f);
        ContextParallel cp_i(mesh, pg, sc, true, RotatorType::AlltoAll, false);
        Tensor out_i = cp_i.forward_cp(qi, ki, vi);
        Tensor ones_i = Tensor::ones(out_i.shape(), opts_ng);
        out_i.backward(&ones_i);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms_fwdbwd = std::chrono::duration<double, std::milli>(t1 - t0).count() / NITERS;
    const double ms_bwd = ms_fwdbwd - ms_fwd;

    // =========================================================================
    // Phase 3 — Output
    //
    // METRIC_MS is the single number the autoresearch tool optimizes.
    //   correct → METRIC_MS = actual fwd+bwd time (lower is better)
    //   broken  → METRIC_MS = 99999.0 (penalty — never accepted)
    // =========================================================================
    if (rank == 0) {
        const double PENALTY = 99999.0;
        const double metric  = correct ? ms_fwdbwd : PENALTY;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "CORRECT=" << (correct ? 1 : 0) << std::endl;
        std::cout << "FWD_MS=" << ms_fwd << std::endl;
        std::cout << "BWD_MS=" << ms_bwd << std::endl;
        std::cout << "METRIC_MS=" << metric << std::endl;

        // Human-readable summary on stderr
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << "\n=== CP Ring Attention Benchmark ===" << std::endl;
        std::cerr << "  Config    : B=" << B << " H=" << H
                  << " T_local=" << Tl << " D=" << D
                  << " (T_full=" << Tl * world_size << ", cp=" << world_size << " GPUs)" << std::endl;
        std::cerr << "  Correct   : " << (correct ? "YES" : "NO") << std::endl;
        std::cerr << "  Forward   : " << ms_fwd << " ms" << std::endl;
        std::cerr << "  Backward  : " << ms_bwd << " ms" << std::endl;
        std::cerr << "  Fwd+Bwd   : " << ms_fwdbwd << " ms" << std::endl;
        std::cerr << "  METRIC    : " << metric << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
