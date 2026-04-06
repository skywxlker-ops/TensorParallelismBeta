// =============================================================================
// CP vs Standard SDPA Numerical Comparison Test
//
// Runs on 2 GPUs with MPI. Computes attention via:
//   (A) Standard path: tril + softmax + matmul (non-CP, full [T x T])
//   (B) CP path: ring attention with SDPAMerger
//
// Compares forward outputs and backward gradients element-by-element
// to pinpoint where CP diverges from the reference.
//
// Build: make cp_sdpa_compare_test
// Run:   mpirun -np 2 ./cp_sdpa_compare_test_exec
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
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

// CP includes
#include "tensor/dtensor.h"
#include "tensor/device_mesh.h"
#include "process_group/ProcessGroupNCCL.h"
#include "gpt2_cp_test/context_parallel/ContextParallel.h"

using namespace OwnTensor;

// Helper: compute max absolute difference between two CPU tensors
float max_abs_diff(const Tensor& a, const Tensor& b) {
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    int64_t n = a.numel();
    float maxd = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        maxd = std::max(maxd, std::abs(pa[i] - pb[i]));
    }
    return maxd;
}

// Helper: compute mean absolute difference
float mean_abs_diff(const Tensor& a, const Tensor& b) {
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    int64_t n = a.numel();
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += std::abs(pa[i] - pb[i]);
    }
    return sum / static_cast<float>(n);
}

// Helper: print first N elements of a CPU tensor
void print_first_n(const Tensor& t, int n, const char* label) {
    const float* p = t.data<float>();
    std::cout << "  " << label << " [first " << n << "]: ";
    for (int i = 0; i < std::min(n, static_cast<int>(t.numel())); ++i) {
        std::cout << std::fixed << std::setprecision(6) << p[i] << " ";
    }
    std::cout << std::endl;
}

// Standard (non-CP) attention forward
// Returns the attention output [B, H, T, D]
Tensor standard_sdpa_forward(
    Tensor& q, Tensor& k, Tensor& v,
    float scale)
{
    // scores = scale * Q @ K^T
    Shape scale_shape({{1}});
    TensorOptions scale_opts = TensorOptions()
        .with_dtype(q.dtype()).with_device(q.device());
    Tensor scale_tensor = Tensor::full(scale_shape, scale_opts, scale);

    Tensor q_scaled = autograd::mul(q, scale_tensor);
    Tensor k_t = autograd::transpose(k, -2, -1);
    Tensor scores = autograd::matmul(q_scaled, k_t);

    // causal mask + softmax (same as gpt2_attn_fixed.cpp)
    float neg_inf = -std::numeric_limits<float>::infinity();
    Tensor masked = autograd::tril(scores, 0, neg_inf);
    Tensor attn_probs = autograd::softmax(masked);

    // output = attn_probs @ v
    Tensor out = autograd::matmul(attn_probs, v);
    return out;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "This test requires exactly 2 GPUs. Run: mpirun -np 2 ./cp_sdpa_compare_test_exec" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    cudaSetDevice(rank);

    // Setup ProcessGroup
    std::vector<int> ranks_vec = {0, 1};
    DeviceMesh mesh({2}, ranks_vec);
    auto pg = mesh.get_process_group(0);

    DeviceIndex device(Device::CUDA, rank);

    // =========================================================================
    // Test parameters — small enough to inspect, large enough to expose bugs
    // =========================================================================
    const int64_t B = 2, H = 2, T = 8, D = 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    Shape qkv_shape({{B, H, T, D}});
    TensorOptions opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(device)
        .with_req_grad(true);

    // Create identical Q, K, V on all ranks (same seed)
    Tensor q_full = Tensor::randn<float>(qkv_shape, opts, 100, 0.5f);
    Tensor k_full = Tensor::randn<float>(qkv_shape, opts, 200, 0.5f);
    Tensor v_full = Tensor::randn<float>(qkv_shape, opts, 300, 0.5f);

    if (rank == 0) {
        std::cout << "=== CP vs Standard SDPA Comparison Test ===" << std::endl;
        std::cout << "Config: B=" << B << " H=" << H << " T=" << T << " D=" << D
                  << " scale=" << scale << " world_size=" << world_size << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // TEST 1: FORWARD COMPARISON
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- TEST 1: Forward Output Comparison ---" << std::endl;
    }

    // --- (A) Standard SDPA on rank 0 ---
    Tensor std_out;
    if (rank == 0) {
        // Clone so we have separate autograd graphs
        Tensor q_std = q_full.clone(); q_std.set_requires_grad(true);
        Tensor k_std = k_full.clone(); k_std.set_requires_grad(true);
        Tensor v_std = v_full.clone(); v_std.set_requires_grad(true);

        std_out = standard_sdpa_forward(q_std, k_std, v_std, scale);
        std::cout << "[Rank 0] Standard SDPA output shape: ";
        std_out.print_meta();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- (B) CP SDPA on both ranks ---
    Tensor q_cp = q_full.clone(); q_cp.set_requires_grad(true);
    Tensor k_cp = k_full.clone(); k_cp.set_requires_grad(true);
    Tensor v_cp = v_full.clone(); v_cp.set_requires_grad(true);

    ContextParallel cp(mesh, pg, scale, /*is_causal=*/true,
                       RotatorType::AlltoAll, /*load_balance=*/false);

    Tensor cp_out = cp.forward_cp(q_cp, k_cp, v_cp);

    if (rank == 0) {
        std::cout << "[Rank 0] CP SDPA output shape: ";
        cp_out.print_meta();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- Compare forward outputs (on rank 0) ---
    if (rank == 0) {
        Tensor std_out_cpu = std_out.to_cpu();
        Tensor cp_out_cpu = cp_out.to_cpu();

        float fwd_max_diff = max_abs_diff(std_out_cpu, cp_out_cpu);
        float fwd_mean_diff = mean_abs_diff(std_out_cpu, cp_out_cpu);

        std::cout << "\n  Forward max  abs diff: " << std::scientific << fwd_max_diff << std::endl;
        std::cout << "  Forward mean abs diff: " << std::scientific << fwd_mean_diff << std::endl;

        if (fwd_max_diff < 1e-4f) {
            std::cout << "  >>> FORWARD PASS: MATCH <<<" << std::endl;
        } else {
            std::cout << "  >>> FORWARD PASS: MISMATCH <<<" << std::endl;
            print_first_n(std_out_cpu, 8, "std_out");
            print_first_n(cp_out_cpu, 8, "cp_out ");

            // Print per-position diff to find where divergence starts
            const float* s = std_out_cpu.data<float>();
            const float* c = cp_out_cpu.data<float>();
            std::cout << "\n  Per-element diff [first 16]:" << std::endl;
            for (int i = 0; i < std::min(16, (int)std_out_cpu.numel()); ++i) {
                float d = std::abs(s[i] - c[i]);
                if (d > 1e-6f) {
                    std::cout << "    [" << i << "] std=" << s[i] << " cp=" << c[i]
                              << " diff=" << d << std::endl;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // TEST 2: BACKWARD COMPARISON
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- TEST 2: Backward Gradient Comparison ---" << std::endl;
    }

    // --- (A) Standard backward on rank 0 ---
    Tensor std_dq, std_dk, std_dv;
    if (rank == 0) {
        Tensor q_std2 = q_full.clone(); q_std2.set_requires_grad(true);
        Tensor k_std2 = k_full.clone(); k_std2.set_requires_grad(true);
        Tensor v_std2 = v_full.clone(); v_std2.set_requires_grad(true);

        Tensor out2 = standard_sdpa_forward(q_std2, k_std2, v_std2, scale);

        // Use ones-filled upstream gradient (equivalent to sum loss)
        Tensor ones_grad = Tensor::full(out2.shape(), out2.opts().with_req_grad(false), 1.0f);
        out2.backward(&ones_grad);

        if (!q_std2.has_grad()) {
            std::cerr << "[Rank 0] ERROR: Standard backward did not produce q gradient!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std_dq = q_std2.grad_view().to_cpu();
        std_dk = k_std2.grad_view().to_cpu();
        std_dv = v_std2.grad_view().to_cpu();

        std::cout << "[Rank 0] Standard backward completed." << std::endl;
        std::cout << "  dQ norm: " << std::sqrt(reduce_sum(std_dq * std_dq, {0,1,2,3}, false).data<float>()[0]) << std::endl;
        std::cout << "  dK norm: " << std::sqrt(reduce_sum(std_dk * std_dk, {0,1,2,3}, false).data<float>()[0]) << std::endl;
        std::cout << "  dV norm: " << std::sqrt(reduce_sum(std_dv * std_dv, {0,1,2,3}, false).data<float>()[0]) << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- (B) CP backward on all ranks ---
    {
        Tensor q_cp2 = q_full.clone(); q_cp2.set_requires_grad(true);
        Tensor k_cp2 = k_full.clone(); k_cp2.set_requires_grad(true);
        Tensor v_cp2 = v_full.clone(); v_cp2.set_requires_grad(true);

        ContextParallel cp2(mesh, pg, scale, /*is_causal=*/true,
                            RotatorType::AlltoAll, /*load_balance=*/false);

        Tensor out_cp2 = cp2.forward_cp(q_cp2, k_cp2, v_cp2);

        // Same ones-filled upstream gradient
        Tensor ones_cp = Tensor::full(out_cp2.shape(), out_cp2.opts().with_req_grad(false), 1.0f);
        out_cp2.backward(&ones_cp);

        if (rank == 0) {
            if (!q_cp2.has_grad()) {
                std::cerr << "[Rank 0] ERROR: CP backward did not produce q gradient!" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            Tensor cp_dq = q_cp2.grad_view().to_cpu();
            Tensor cp_dk = k_cp2.grad_view().to_cpu();
            Tensor cp_dv = v_cp2.grad_view().to_cpu();

            std::cout << "\n[Rank 0] CP backward completed." << std::endl;
            std::cout << "  dQ norm: " << std::sqrt(reduce_sum(cp_dq * cp_dq, {0,1,2,3}, false).data<float>()[0]) << std::endl;
            std::cout << "  dK norm: " << std::sqrt(reduce_sum(cp_dk * cp_dk, {0,1,2,3}, false).data<float>()[0]) << std::endl;
            std::cout << "  dV norm: " << std::sqrt(reduce_sum(cp_dv * cp_dv, {0,1,2,3}, false).data<float>()[0]) << std::endl;

            // Compare against standard
            float dq_max = max_abs_diff(std_dq, cp_dq);
            float dk_max = max_abs_diff(std_dk, cp_dk);
            float dv_max = max_abs_diff(std_dv, cp_dv);
            float dq_mean = mean_abs_diff(std_dq, cp_dq);
            float dk_mean = mean_abs_diff(std_dk, cp_dk);
            float dv_mean = mean_abs_diff(std_dv, cp_dv);

            std::cout << "\n  Gradient comparison (std vs CP):" << std::endl;
            std::cout << "  dQ: max_diff=" << std::scientific << dq_max << " mean_diff=" << dq_mean << std::endl;
            std::cout << "  dK: max_diff=" << std::scientific << dk_max << " mean_diff=" << dk_mean << std::endl;
            std::cout << "  dV: max_diff=" << std::scientific << dv_max << " mean_diff=" << dv_mean << std::endl;

            float tol = 1e-3f;
            bool pass = (dq_max < tol && dk_max < tol && dv_max < tol);
            if (pass) {
                std::cout << "\n  >>> BACKWARD PASS: MATCH (tol=" << tol << ") <<<" << std::endl;
            } else {
                std::cout << "\n  >>> BACKWARD PASS: MISMATCH <<<" << std::endl;

                // Print which gradient diverges most
                if (dq_max >= tol) {
                    std::cout << "\n  dQ diverges — first mismatches:" << std::endl;
                    const float* sq = std_dq.data<float>();
                    const float* cq = cp_dq.data<float>();
                    int printed = 0;
                    for (int i = 0; i < (int)std_dq.numel() && printed < 10; ++i) {
                        float d = std::abs(sq[i] - cq[i]);
                        if (d > 1e-5f) {
                            // Decode index to [b, h, t, d]
                            int idx = i;
                            int d_dim = idx % D; idx /= D;
                            int t_dim = idx % T; idx /= T;
                            int h_dim = idx % H; idx /= H;
                            int b_dim = idx;
                            std::cout << "    [b=" << b_dim << ",h=" << h_dim
                                      << ",t=" << t_dim << ",d=" << d_dim
                                      << "] std=" << sq[i] << " cp=" << cq[i]
                                      << " diff=" << d << std::endl;
                            printed++;
                        }
                    }
                }
                if (dk_max >= tol) {
                    std::cout << "\n  dK diverges — first mismatches:" << std::endl;
                    const float* sk = std_dk.data<float>();
                    const float* ck = cp_dk.data<float>();
                    int printed = 0;
                    for (int i = 0; i < (int)std_dk.numel() && printed < 10; ++i) {
                        float d = std::abs(sk[i] - ck[i]);
                        if (d > 1e-5f) {
                            int idx = i;
                            int d_dim = idx % D; idx /= D;
                            int t_dim = idx % T; idx /= T;
                            int h_dim = idx % H; idx /= H;
                            int b_dim = idx;
                            std::cout << "    [b=" << b_dim << ",h=" << h_dim
                                      << ",t=" << t_dim << ",d=" << d_dim
                                      << "] std=" << sk[i] << " cp=" << ck[i]
                                      << " diff=" << d << std::endl;
                            printed++;
                        }
                    }
                }
                if (dv_max >= tol) {
                    std::cout << "\n  dV diverges — first mismatches:" << std::endl;
                    const float* sv = std_dv.data<float>();
                    const float* cv = cp_dv.data<float>();
                    int printed = 0;
                    for (int i = 0; i < (int)std_dv.numel() && printed < 10; ++i) {
                        float d = std::abs(sv[i] - cv[i]);
                        if (d > 1e-5f) {
                            int idx = i;
                            int d_dim = idx % D; idx /= D;
                            int t_dim = idx % T; idx /= T;
                            int h_dim = idx % H; idx /= H;
                            int b_dim = idx;
                            std::cout << "    [b=" << b_dim << ",h=" << h_dim
                                      << ",t=" << t_dim << ",d=" << d_dim
                                      << "] std=" << sv[i] << " cp=" << cv[i]
                                      << " diff=" << d << std::endl;
                            printed++;
                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // TIMING: Full CP ring-attention vs Megatron TEDotProductAttention
    //
    // Each rank holds T_local = T/2 tokens.  forward_cp runs the full ring
    // attention: 2 ring steps (local SDPA + NCCL K/V rotation each) + merge.
    // Equivalent to what Megatron's TEDotProductAttention does on 2 GPUs.
    //
    // Megatron reference: B=4 H=6 T_local=512 D=64, t_attn total=49.47ms
    //   over 48 calls (16 grad_accum x 3 layers) => 1.031 ms per call.
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- TIMING: Full CP ring-attn vs Megatron TEDotProductAttn ---" << std::endl;
    }

    {
        // Use GPT-2 medium config: B=4, H=6, T_local=512 (T_full=1024), D=64
        const int64_t Bt = 4, Ht = 6, Tl = 512, Dt = 64;
        const float   sc = 1.0f / std::sqrt(static_cast<float>(Dt));
        const int     NWARM = 5, NITERS = 20;

        Shape qkv_t({{Bt, Ht, Tl, Dt}});
        TensorOptions opts_t = TensorOptions()
            .with_dtype(Dtype::Float32)
            .with_device(device)
            .with_req_grad(false);

        Tensor qt = Tensor::randn<float>(qkv_t, opts_t, 77, 0.3f);
        Tensor kt = Tensor::randn<float>(qkv_t, opts_t, 78, 0.3f);
        Tensor vt = Tensor::randn<float>(qkv_t, opts_t, 79, 0.3f);

        // Warm-up
        for (int w = 0; w < NWARM; ++w) {
            ContextParallel cp_w(mesh, pg, sc, true, RotatorType::AlltoAll, false);
            cp_w.forward_cp(qt, kt, vt);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Timed run
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NITERS; ++i) {
            ContextParallel cp_i(mesh, pg, sc, true, RotatorType::AlltoAll, false);
            cp_i.forward_cp(qt, kt, vt);
        }
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();

        const double ms_cp = std::chrono::duration<double, std::milli>(t1 - t0).count() / NITERS;

        const double megatron_ms = 49.478 / (16 * 3);  // per call

        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Config                      : B=" << Bt
                      << " H=" << Ht << " T_local=" << Tl << " D=" << Dt
                      << " (T_full=" << Tl * world_size << ", cp=" << world_size << " GPUs)" << std::endl;
            std::cout << "  Our CP forward_cp (FP32)    : " << ms_cp
                      << " ms  [full ring-attn: " << world_size
                      << "x SDPA + " << world_size << "x NCCL K/V comm + merge]" << std::endl;
            std::cout << "  Megatron TEDotProduct (BF16): " << megatron_ms
                      << " ms  [full ring-attn: 2x SDPA + 2x ring comm, cuDNN FA]" << std::endl;
            std::cout << "  Megatron speedup vs ours    : "
                      << (ms_cp / megatron_ms) << "x" << std::endl;
            std::cout << "  (Megatron uses BF16 tensor cores + cuDNN FlashAttn; ours is FP32 from scratch)" << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // =========================================================================
    // TIMING: Full CP forward + backward vs Megatron forward-only
    //
    // Megatron's t_attn timer wraps only the TEDotProductAttention forward call
    // (lines 204-214 in megatronCP.py).  Backward is not included.
    // So for a fair end-to-end comparison we also time our full fwd+bwd.
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- TIMING: Full CP fwd+bwd (our FP32) ---" << std::endl;
    }

    {
        const int64_t Bt = 4, Ht = 6, Tl = 512, Dt = 64;
        const float   sc = 1.0f / std::sqrt(static_cast<float>(Dt));
        const int     NWARM = 3, NITERS = 20;

        Shape qkv_t({{Bt, Ht, Tl, Dt}});
        TensorOptions opts_rg = TensorOptions()
            .with_dtype(Dtype::Float32)
            .with_device(device)
            .with_req_grad(true);
        TensorOptions opts_ng = opts_rg.with_req_grad(false);

        // Warm-up
        for (int w = 0; w < NWARM; ++w) {
            Tensor qw = Tensor::randn<float>(qkv_t, opts_rg, 90 + w, 0.3f);
            Tensor kw = Tensor::randn<float>(qkv_t, opts_rg, 91 + w, 0.3f);
            Tensor vw = Tensor::randn<float>(qkv_t, opts_rg, 92 + w, 0.3f);
            ContextParallel cp_w(mesh, pg, sc, true, RotatorType::AlltoAll, false);
            Tensor out_w = cp_w.forward_cp(qw, kw, vw);
            Tensor ones_w = Tensor::ones(out_w.shape(), opts_ng);
            out_w.backward(&ones_w);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Timed forward-only
        auto tf0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NITERS; ++i) {
            Tensor qi = Tensor::randn<float>(qkv_t, opts_ng, 200 + i, 0.3f);
            Tensor ki = Tensor::randn<float>(qkv_t, opts_ng, 201 + i, 0.3f);
            Tensor vi = Tensor::randn<float>(qkv_t, opts_ng, 202 + i, 0.3f);
            ContextParallel cp_i(mesh, pg, sc, true, RotatorType::AlltoAll, false);
            cp_i.forward_cp(qi, ki, vi);
        }
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto tf1 = std::chrono::high_resolution_clock::now();
        const double ms_fwd_only = std::chrono::duration<double, std::milli>(tf1 - tf0).count() / NITERS;

        // Timed fwd+bwd
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NITERS; ++i) {
            Tensor qi = Tensor::randn<float>(qkv_t, opts_rg, 100 + i, 0.3f);
            Tensor ki = Tensor::randn<float>(qkv_t, opts_rg, 101 + i, 0.3f);
            Tensor vi = Tensor::randn<float>(qkv_t, opts_rg, 102 + i, 0.3f);
            ContextParallel cp_i(mesh, pg, sc, true, RotatorType::AlltoAll, false);
            Tensor out_i = cp_i.forward_cp(qi, ki, vi);
            Tensor ones_i = Tensor::ones(out_i.shape(), opts_ng);
            out_i.backward(&ones_i);
        }
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();

        const double ms_fwdbwd = std::chrono::duration<double, std::milli>(t1 - t0).count() / NITERS;

        // Run Megatron fwd+bwd bench on rank 0 only (torchrun spawns its own processes)
        double meg_fwd_ms = -1.0, meg_bwd_ms = -1.0;
        if (rank == 0) {
            const char* bench_cmd =
                "torchrun --nproc_per_node=2 "
                "/home/blu-bridge25/TP/TensorParallelismBeta/Megatron-LM/megatron_attn_bench.py "
                "2>/dev/null";
            FILE* pipe = popen(bench_cmd, "r");
            if (pipe) {
                char line[128];
                while (fgets(line, sizeof(line), pipe)) {
                    std::string s(line);
                    if (s.rfind("FWD_MS=", 0) == 0)
                        meg_fwd_ms = std::stod(s.substr(7));
                    else if (s.rfind("BWD_MS=", 0) == 0)
                        meg_bwd_ms = std::stod(s.substr(7));
                }
                pclose(pipe);
            }
        }

        // Run PyTorch fwd+bwd bench on rank 0 only
        double pt_fwd_ms = -1.0, pt_fwdbwd = -1.0;
        if (rank == 0) {
            const char* pt_bench_cmd =
                "python3 /home/blu-bridge25/TP/TensorParallelismBeta/DTensor/pytorch_attn_bench.py 2>/dev/null";
            FILE* pipe = popen(pt_bench_cmd, "r");
            if (pipe) {
                char line[128];
                double bwd = 0.0;
                while (fgets(line, sizeof(line), pipe)) {
                    std::string s(line);
                    if (s.rfind("PT_FWD_MS=", 0) == 0)
                        pt_fwd_ms = std::stod(s.substr(10));
                    else if (s.rfind("PT_BWD_MS=", 0) == 0)
                        bwd = std::stod(s.substr(10));
                }
                pclose(pipe);
                if (pt_fwd_ms > 0 && bwd > 0) pt_fwdbwd = pt_fwd_ms + bwd;
            }
        }

        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Config                       : B=" << Bt
                      << " H=" << Ht << " T_local=" << Tl << " D=" << Dt
                      << " (T_full=" << Tl * world_size << ", cp=" << world_size << " GPUs)" << std::endl;
            std::cout << std::endl;
            std::cout << "                                  Our C++ (TF32)  PyTorch (TF32)  Megatron (BF16, cuDNN FA)" << std::endl;
            std::cout << "  Forward only                :  " << std::setw(7) << ms_fwd_only
                      << " ms      " << std::setw(7) << pt_fwd_ms << " ms";
            if (meg_fwd_ms > 0) std::cout << std::setw(7) << meg_fwd_ms << " ms";
            else                 std::cout << "  N/A";
            std::cout << std::endl;
            std::cout << "  Backward only               :  " << std::setw(7) << (ms_fwdbwd - ms_fwd_only)
                      << " ms      " << std::setw(7) << (pt_fwdbwd - pt_fwd_ms) << " ms";
            if (meg_bwd_ms > 0) std::cout << std::setw(7) << meg_bwd_ms << " ms";
            else                 std::cout << "  N/A";
            std::cout << std::endl;
            std::cout << "  Forward + Backward          :  " << std::setw(7) << ms_fwdbwd
                      << " ms      " << std::setw(7) << pt_fwdbwd << " ms";
            if (meg_fwd_ms > 0 && meg_bwd_ms > 0)
                std::cout << std::setw(7) << (meg_fwd_ms + meg_bwd_ms) << " ms";
            else
                std::cout << "  N/A";
            std::cout << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::cout << "\n=== Test Complete ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
