// =============================================================================
// CP forward-overlap co-execution race repro (2 GPUs, MPI)
//
// Isolates the hypothesis: does the fused-attention forward kernel produce a
// DIFFERENT output when it co-executes with a concurrent NCCL ring send/recv
// (the forward-overlap condition) vs running alone?  No training loop, no
// merger, no backward, no recompute-state confound -- just:
//
//   out_alone = sdpa_fused_forward(Q,K,V)            // nothing else on the GPU
//   out_conc  = sdpa_fused_forward(Q,K,V)            // while NCCL sendrecv runs
//                                                    //   on cp_ring_stream_
//   compare out_alone vs out_conc, looped many times
//
// Same fixed Q/K/V every iteration, so the kernel is deterministic => any
// nonzero diff means co-execution changes the SDPA output (forward corruption).
// Bit-equal over many iters => co-execution does NOT corrupt the SDPA, and the
// training divergence is elsewhere (backward).
//
// Build: make cp_coexec_race_test
// Run:   mpirun -np 2 ./cp_coexec_race_test_exec
//        NITERS=5000 RING_BURST=8 mpirun -np 2 ./cp_coexec_race_test_exec
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

#include "TensorLib.h"
#include "tensor/dtensor.h"
#include "tensor/device_mesh.h"
#include "process_group/ProcessGroupNCCL.h"
#include "gpt2_cp_test/context_parallel/SDPAOp.h"
#include "gpt2_cp_test/context_parallel/FusedSDPAOp.h"

using namespace OwnTensor;

static double max_abs_diff_host(const Tensor &a, const Tensor &b) {
  Tensor ha = a.to_cpu();
  Tensor hb = b.to_cpu();
  const float *pa = ha.data<float>();
  const float *pb = hb.data<float>();
  int64_t n = std::min<int64_t>(ha.numel(), hb.numel());
  double mx = 0.0;
  for (int64_t j = 0; j < n; ++j) {
    double d = std::fabs((double)pa[j] - (double)pb[j]);
    if (d > mx) mx = d;
  }
  return mx;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size != 2) {
    if (rank == 0)
      std::cerr << "Requires exactly 2 GPUs: mpirun -np 2 ./cp_coexec_race_test_exec\n";
    MPI_Finalize();
    return 1;
  }
  cudaSetDevice(rank);

  std::vector<int> ranks_vec = {0, 1};
  DeviceMesh mesh({2}, ranks_vec);
  auto pg = mesh.get_process_group(0);
  DeviceIndex device(Device::CUDA, rank);

  // Per-rank chunk shape for the 44M config, ws=2: B=4 H=6 T_local=512 D=64.
  const int64_t B = 4, H = 6, T = 512, D = 64;
  const float scale = 1.0f / std::sqrt((float)D);
  const bool is_causal = true; // matches the i==0 ring step (local causal SDPA)

  const int NITERS = std::getenv("NITERS") ? std::atoi(std::getenv("NITERS")) : 3000;
  const int RING_BURST =
      std::getenv("RING_BURST") ? std::atoi(std::getenv("RING_BURST")) : 8;

  TensorOptions opts = TensorOptions()
                           .with_dtype(Dtype::Float32)
                           .with_device(device)
                           .with_req_grad(false);

  // Fixed inputs (same every iteration). Distinct seeds per tensor; same on
  // both ranks is fine -- the ring transfer content is irrelevant, we only care
  // whether the concurrent NCCL kernel perturbs the SDPA output.
  Shape qkv({{B, H, T, D}});
  Tensor Q = Tensor::randn<float>(qkv, opts, 11, 0.3f);
  Tensor K = Tensor::randn<float>(qkv, opts, 22, 0.3f);
  Tensor V = Tensor::randn<float>(qkv, opts, 33, 0.3f);

  // Ring staging: send_buf holds K then V (kv_numel), recv_buf receives.
  int64_t k_numel = K.numel();
  int64_t kv_numel = k_numel * 2;
  Tensor send_buf = Tensor::empty(Shape({{kv_numel}}), opts);
  Tensor recv_buf = Tensor::empty(Shape({{kv_numel}}), opts);
  cudaMemcpyAsync(send_buf.data<float>(), K.data<float>(),
                  k_numel * sizeof(float), cudaMemcpyDeviceToDevice, 0);
  cudaMemcpyAsync(send_buf.data<float>() + k_numel, V.data<float>(),
                  k_numel * sizeof(float), cudaMemcpyDeviceToDevice, 0);
  cudaStreamSynchronize(0);

  const int next_rank = (rank + 1) % world_size;
  const int prev_rank = (rank - 1 + world_size) % world_size;
  cudaStream_t rs = pg->cpRingStream();

  if (rank == 0) {
    std::cout << "[co-exec repro] shape=[" << B << "," << H << "," << T << ","
              << D << "] causal=" << is_causal << " NITERS=" << NITERS
              << " RING_BURST=" << RING_BURST << "\n";
  }

  // Reference: SDPA with NOTHING co-running on the GPU.
  cudaDeviceSynchronize();
  SDPAResult ref = sdpa_fused_forward(Q, K, V, is_causal, scale, 0, 0);
  cudaDeviceSynchronize();
  Tensor h_ref_out = ref.out.to_cpu();
  Tensor h_ref_lse = ref.lse.to_cpu();

  double worst_out = 0.0, worst_lse = 0.0;
  int first_bad_iter = -1;
  double first_bad_out = 0.0, first_bad_lse = 0.0;

  for (int it = 0; it < NITERS; ++it) {
    // ---- (A) alone: confirm determinism (should always match ref) ----
    cudaDeviceSynchronize();
    SDPAResult a = sdpa_fused_forward(Q, K, V, is_causal, scale, 0, 0);
    cudaDeviceSynchronize();
    double da_out = max_abs_diff_host(a.out, h_ref_out);
    double da_lse = max_abs_diff_host(a.lse, h_ref_lse);

    // ---- (B) concurrent: keep the ring stream busy while the SDPA runs ----
    // Post a burst of sendrecv on the ring stream (async, not waited), then
    // immediately launch the SDPA on the default stream so they co-execute.
    std::shared_ptr<Work> w;
    for (int r = 0; r < RING_BURST; ++r) {
      w = pg->sendrecv_async_stream(send_buf.data<float>(),
                                    recv_buf.data<float>(), next_rank,
                                    prev_rank, (size_t)kv_numel,
                                    Dtype::Float32, rs);
    }
    SDPAResult c = sdpa_fused_forward(Q, K, V, is_causal, scale, 0, 0);
    if (w) w->wait();
    cudaStreamSynchronize(rs);
    cudaStreamSynchronize(0);
    cudaDeviceSynchronize();
    double dc_out = max_abs_diff_host(c.out, h_ref_out);
    double dc_lse = max_abs_diff_host(c.lse, h_ref_lse);

    if (dc_out > worst_out) worst_out = dc_out;
    if (dc_lse > worst_lse) worst_lse = dc_lse;
    if ((dc_out > 0.0 || dc_lse > 0.0) && first_bad_iter < 0) {
      first_bad_iter = it;
      first_bad_out = dc_out;
      first_bad_lse = dc_lse;
    }
    // Sanity: the alone path must stay deterministic; flag loudly if not.
    if ((da_out > 0.0 || da_lse > 0.0) && rank == 0) {
      std::cout << "[co-exec repro] WARNING alone-path nondeterministic at it="
                << it << " out=" << da_out << " lse=" << da_lse << "\n";
    }
    if (rank == 0 && (it % 500 == 0)) {
      std::cout << "[co-exec repro] it=" << it << " conc_out_diff=" << dc_out
                << " conc_lse_diff=" << dc_lse << "\n";
    }
  }

  if (rank == 0) {
    std::cout << "\n[co-exec repro] DONE rank0\n"
              << "  worst conc out_diff = " << worst_out << "\n"
              << "  worst conc lse_diff = " << worst_lse << "\n";
    if (first_bad_iter >= 0)
      std::cout << "  FIRST nonzero conc diff at it=" << first_bad_iter
                << " out=" << first_bad_out << " lse=" << first_bad_lse
                << "  => CO-EXECUTION CORRUPTS THE SDPA OUTPUT\n";
    else
      std::cout << "  conc output bit-identical to ref across all iters"
                << "  => co-execution does NOT corrupt the SDPA (look at backward)\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
