// Isolated CP backward pass test (C++ side).
//
// Loads Q_local, K_local, V_local, dY_local from .bin files produced by
// Pytorch/cp_bwd_isolated_test.py, runs ONE CP forward + backward, and
// dumps per-step grads via the DUMP_CP_STEPS=1 instrumentation already
// present in ContextParallelBackward.h. Saves final dQ, dK, dV to .bin
// for diff against the PT side.
//
// Run:
//   DUMP_CP_STEPS=1 mpirun -np 2 ./cp_bwd_isolated_test_exec

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/dtensor.h"

#include "gpt2_cp_test/context_parallel/ContextParallel.h"

using namespace OwnTensor;

namespace {

constexpr int B = 1, H = 2, T = 128, D = 64;
const std::string DUMP_DIR = "/tmp/cp_bwd_test";

Tensor load_bin(const std::string &path, const Shape &shape, DeviceIndex device) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    throw std::runtime_error("cannot open " + path);
  }
  TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);
  Tensor host = Tensor::empty(shape, cpu_opts);
  int64_t bytes = host.numel() * sizeof(float);
  fin.read(reinterpret_cast<char *>(host.data<float>()), bytes);
  if (!fin) {
    throw std::runtime_error("short read on " + path);
  }
  return host.to(device);
}

void save_bin(const Tensor &t, const std::string &path) {
  Tensor host = t.to_cpu();
  std::ofstream fout(path, std::ios::binary);
  int64_t bytes = host.numel() * sizeof(float);
  fout.write(reinterpret_cast<const char *>(host.data<float>()), bytes);
}

void dump_first16(std::ostream &os, const char *label, const Tensor &t) {
  Tensor host = t.to_cpu();
  const float *p = host.data<float>();
  const auto &dims = t.shape().dims;
  os << label << " shape=[";
  for (size_t i = 0; i < dims.size(); ++i) {
    os << dims[i] << (i + 1 == dims.size() ? "" : ",");
  }
  os << "] first16: [";
  int64_t n = std::min<int64_t>(16, t.numel());
  for (int64_t i = 0; i < n; ++i) {
    os << p[i] << (i == n - 1 ? "" : ", ");
  }
  os << "]\n";
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size != 2) {
    if (rank == 0) std::cerr << "This test expects world_size=2\n";
    MPI_Finalize();
    return 1;
  }
  cudaSetDevice(rank);
  DeviceIndex device(Device::CUDA, rank);

  // Ensure dump dirs exist (rank 0 creates; other ranks barrier below).
  if (rank == 0) {
    std::filesystem::create_directories("/tmp/cp_bwd_test/deep");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Clear stale per-step dump files from previous runs (the dump in
  // ContextParallelBackward.h opens with std::ios::app and otherwise
  // accumulates across the lifetime of all binaries, making diffs against
  // the PT isolated test misleading).
  {
    std::string p = "step_bw_rank" + std::to_string(rank) + ".md";
    std::ofstream(p, std::ios::trunc);
  }

  const int64_t T_local = T / world_size;

  // ProcessGroup + DeviceMesh
  auto pg = init_process_group(world_size, rank);
  std::vector<int> ranks_vec(world_size);
  for (int r = 0; r < world_size; ++r) ranks_vec[r] = r;
  DeviceMesh mesh({world_size}, ranks_vec);

  // Load per-rank inputs from .bin
  Shape local_shape({{B, H, T_local, D}});
  std::string base = DUMP_DIR + "/";
  Tensor Q_local = load_bin(base + "Q_local_rank" + std::to_string(rank) + ".bin",
                             local_shape, device);
  Tensor K_local = load_bin(base + "K_local_rank" + std::to_string(rank) + ".bin",
                             local_shape, device);
  Tensor V_local = load_bin(base + "V_local_rank" + std::to_string(rank) + ".bin",
                             local_shape, device);
  Tensor dY_local = load_bin(base + "dY_local_rank" + std::to_string(rank) + ".bin",
                              local_shape, device);

  // Sanity dump — write first16 of each input so we can confirm bit-for-bit
  // parity with the PT-side test that produced these .bin files.
  {
    std::ofstream sf(base + "cpp_inputs_rank" + std::to_string(rank) + ".md");
    dump_first16(sf, "Q_local", Q_local);
    dump_first16(sf, "K_local", K_local);
    dump_first16(sf, "V_local", V_local);
    dump_first16(sf, "dY_local", dY_local);
  }

  // ===== Verify: PT-permuted Q_local matches what C++'s chunk-level HeadTail
  // formula (used by shard_sequence_pre_embed in production training) would
  // produce when applied to the full Q.
  //
  // C++ formula for rank r, N world_size, chunk_sz = T/(2N):
  //   local[0..chunk_sz-1]   = full[r*chunk_sz             .. (r+1)*chunk_sz-1]   (head_chunk)
  //   local[chunk_sz..T_local-1] = full[(2N-1-r)*chunk_sz .. (2N-r)*chunk_sz-1] (tail_chunk)
  {
    Shape full_shape({{B, H, T, D}});
    Tensor Q_full = load_bin(base + "Q_full.bin", full_shape, device);
    int64_t chunk_sz = T / (2 * world_size);
    int64_t head_chunk = rank;
    int64_t tail_chunk = 2 * world_size - 1 - rank;
    // Build expected Q_local on CPU by gathering rows from Q_full.
    Tensor Q_full_cpu = Q_full.to_cpu();
    TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);
    Tensor expected_local_cpu = Tensor::empty(local_shape, cpu_opts);
    const float *src = Q_full_cpu.data<float>();
    float *dst = expected_local_cpu.data<float>();
    int64_t per_T = D;
    int64_t per_H = T * D;          // stride of one head in full
    int64_t per_H_local = T_local * D;  // stride of one head in local
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        const float *src_bh = src + b * H * T * D + h * per_H;
        float *dst_bh = dst + b * H * T_local * D + h * per_H_local;
        // head_chunk -> local positions 0..chunk_sz-1
        std::memcpy(dst_bh,
                    src_bh + head_chunk * chunk_sz * per_T,
                    chunk_sz * per_T * sizeof(float));
        // tail_chunk -> local positions chunk_sz..2*chunk_sz-1
        std::memcpy(dst_bh + chunk_sz * per_T,
                    src_bh + tail_chunk * chunk_sz * per_T,
                    chunk_sz * per_T * sizeof(float));
      }
    }
    Tensor Q_loaded_cpu = Q_local.to_cpu();
    const float *loaded = Q_loaded_cpu.data<float>();
    const float *expected = expected_local_cpu.data<float>();
    float max_ad = 0.0f;
    int64_t arg_max = -1;
    int64_t n = Q_local.numel();
    for (int64_t i = 0; i < n; ++i) {
      float d = std::abs(loaded[i] - expected[i]);
      if (d > max_ad) { max_ad = d; arg_max = i; }
    }
    if (rank == 0) {
      std::cout << "[perm parity] PT_Q_local vs C++-formula(Q_full) max_abs_diff="
                << max_ad << (max_ad == 0.0f ? "  bit-exact match"
                                               : "  MISMATCH (perm formulas differ!)")
                << "\n";
      if (max_ad > 0.0f && arg_max >= 0) {
        std::cout << "  argmax flat index=" << arg_max
                  << "  loaded=" << loaded[arg_max]
                  << "  expected=" << expected[arg_max] << "\n";
      }
    }
  }

  Q_local.set_requires_grad(true);
  K_local.set_requires_grad(true);
  V_local.set_requires_grad(true);

  // LB toggle: set CP_LB=0 to run without HeadTail load balancing.
  bool use_lb = true;
  if (const char *env = std::getenv("CP_LB")) {
    use_lb = (env[0] != '0');
  }
  if (rank == 0) {
    std::cout << "[rank 0] CP_LB=" << (use_lb ? 1 : 0)
              << " (load_balance " << (use_lb ? "ON" : "OFF") << ")\n";
  }

  // ContextParallel module
  float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));
  ContextParallel cp(mesh, pg, attn_scale,
                     /*is_causal=*/true,
                     /*rotator_type=*/RotatorType::P2P,
                     /*load_balance=*/use_lb,
                     /*recompute_k=*/false);

  // forward_cp expects Q/K/V already in HeadTail layout per rank
  // (pre_sharded=true means CP won't apply its own HeadTail kernel).
  Tensor output = cp.forward_cp(Q_local, K_local, V_local,
                                /*unshard=*/false,
                                /*pre_sharded=*/true);

  if (rank == 0) {
    std::cout << "[rank 0] forward_cp done. output shape=["
              << output.shape().dims[0] << "," << output.shape().dims[1] << ","
              << output.shape().dims[2] << "," << output.shape().dims[3]
              << "]\n";
  }

  save_bin(output, base + "out_cpp_rank" + std::to_string(rank) + ".bin");

  // Trigger backward by passing dY_local as the upstream gradient at `output`.
  autograd::backward(output, &dY_local);

  // Collect grads
  Tensor dQ = Q_local.grad_view();
  Tensor dK = K_local.grad_view();
  Tensor dV = V_local.grad_view();

  save_bin(dQ, base + "dQ_cpp_rank" + std::to_string(rank) + ".bin");
  save_bin(dK, base + "dK_cpp_rank" + std::to_string(rank) + ".bin");
  save_bin(dV, base + "dV_cpp_rank" + std::to_string(rank) + ".bin");

  // Also write a sanity dump (first-16 of dQ/dK/dV) to a human-readable file
  std::string sanity_path = base + "cpp_final_rank" + std::to_string(rank) + ".md";
  std::ofstream sf(sanity_path);
  dump_first16(sf, "dQ", dQ);
  dump_first16(sf, "dK", dK);
  dump_first16(sf, "dV", dV);

  if (rank == 0) {
    std::cout << "[rank 0] backward done. Outputs written to " << DUMP_DIR << "\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
