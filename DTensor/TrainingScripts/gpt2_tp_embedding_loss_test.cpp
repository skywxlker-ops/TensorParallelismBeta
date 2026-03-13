#include "dnn/DistributedNN.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/dtensor.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpi.h>

using namespace dnn;
using namespace autograd;
using namespace OwnTensor;

bool allclose(const Tensor &a, const Tensor &b, float rtol = 1e-4,
              float atol = 1e-4) {
  auto cpu_a = a.to_cpu();
  auto cpu_b = b.to_cpu();
  const float *p_a = cpu_a.data<float>();
  const float *p_b = cpu_b.data<float>();
  int64_t n = cpu_a.numel();

  if (n != cpu_b.numel())
    return false;

  float max_diff = 0;
  for (int64_t i = 0; i < n; ++i) {
    float diff = std::abs(p_a[i] - p_b[i]);
    if (diff > max_diff)
      max_diff = diff;
    if (diff > (atol + rtol * std::abs(p_b[i]))) {
      std::cout << "Mismatch at index " << i << ": " << p_a[i]
                << " != " << p_b[i] << std::endl;
      return false;
    }
  }
  std::cout << "Max diff: " << max_diff << std::endl;
  return true;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  OwnTensor::DeviceIndex device(OwnTensor::Device::CUDA, rank);
  cudaSetDevice(rank);

  std::vector<int> ranks_vec(world_size);
  for (int i = 0; i < world_size; ++i)
    ranks_vec[i] = i;
  DeviceMesh mesh({world_size}, ranks_vec);
  auto pg = mesh.get_process_group(0);

  if (rank == 0)
    std::cout << "=== TENSOR PARALLEL EMBEDDING & LOSS EQUIVALENCE TEST ==="
              << std::endl;

  // --- Configurations ---
  int64_t B = 2;
  int64_t T = 4;
  int64_t C = 8;
  int64_t V = 12; // Small vocab for easy debugging

  // --- Ground Truth Data (Identical on all ranks) ---
  TensorOptions opts =
      TensorOptions().with_dtype(Dtype::Float32).with_device(device);
  TensorOptions idx_opts =
      TensorOptions().with_dtype(Dtype::Int64).with_device(device);

  // 1. Inputs & Targets
  // B=2, T=4
  std::vector<int64_t> input_ids = {
      0, 1, 5, 11, // Batch 0
      2, 3, 7, 10  // Batch 1
  };
  Tensor input_tensor = Tensor::zeros(Shape{{B * T}}, idx_opts);
  cudaMemcpy(input_tensor.data<int64_t>(), input_ids.data(),
             input_ids.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  input_tensor = autograd::reshape(input_tensor, Shape{{B, T}});

  std::vector<int64_t> target_ids = {
      1, 5, 11, 2, // Batch 0
      3, 7, 10, 0  // Batch 1
  };
  Tensor target_tensor = Tensor::zeros(Shape{{B * T}}, idx_opts);
  cudaMemcpy(target_tensor.data<int64_t>(), target_ids.data(),
             target_ids.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  target_tensor = autograd::reshape(target_tensor, Shape{{B, T}});

  // 2. Full Embedding Weight [V, C]
  Tensor full_weight = Tensor::randn<float>(Shape{{V, C}}, opts, 42, 1.0f);
  full_weight.set_requires_grad(true);

  // =========================================================================
  // PART 1: Non-Distributed (Serial) Reference (Computed only on Rank 0 but
  // let's do both for simplicity)
  // =========================================================================

  // 1a. Serial Embedding
  Tensor serial_out = autograd::embedding(full_weight, input_tensor);
  serial_out.set_requires_grad(true);

  // 1b. Mock Network
  Tensor serial_logits = serial_out;
  // serial_logits = serial_logits.as_type(Dtype::Float32); // removed

  // Reshape to 2D for matmul: [B*T, C]
  if (rank == 0) {
    auto dims = serial_logits.shape().dims;
    std::cout << "serial_logits dimensions: ";
    for (auto d : dims)
      std::cout << d << " ";
    std::cout << "\nnumel: " << serial_logits.numel() << std::endl;
    std::cout << "B*T: " << B * T << ", C: " << C
              << ", expected numel: " << (B * T * C) << std::endl;
  }
  Tensor serial_logits_flat =
      autograd::reshape(serial_logits, Shape{{B * T, C}});

  // 1c. Serial Loss (PyTorch style cross entropy over [B*T, V] with [B*T]
  // targets) Create LM Head weights (just use embedding weights transposed)
  Tensor head_weights = full_weight.t(); // [C, V]
  // [B*T, C] x [C, V] -> [B*T, V]
  Tensor serial_full_logits_flat =
      OwnTensor::autograd::matmul(serial_logits_flat, head_weights);

  // Targets
  Tensor serial_targets_1d = autograd::reshape(target_tensor, Shape{{B * T}});

  Tensor serial_loss = OwnTensor::autograd::sparse_cross_entropy_loss(
      serial_full_logits_flat, serial_targets_1d);

  // Backward
  serial_loss.backward();

  if (rank == 0) {
    std::cout << "\n[Serial] Loss: " << serial_loss.to_cpu().data<float>()[0]
              << std::endl;
    std::cout << "[Serial] Embedding Weight Grad (First row): \n";
    if (full_weight.has_grad()) {
      Tensor gw = full_weight.grad_view().to_cpu();
      for (int i = 0; i < C; ++i)
        std::cout << gw.data<float>()[i] << " ";
    } else {
      std::cout << "GRADIENT NOT ALLOCATED";
    }
    std::cout << "\n";
  }

  // =========================================================================
  // PART 2: Distributed (Vocab Parallel) Implementation
  // =========================================================================

  // 2a. Distributed Embedding
  // Let's pass the 1.0f and 42 correctly matching the signature:
  // DEmbeddingVParallelFused(mesh, pg, vocab_size, embedding_dim, sd, seed)
  DEmbeddingVParallelFused dist_emb(mesh, pg, V, C, 0.02f, 42);

  int64_t v_local = V / world_size;
  int64_t v_start = rank * v_local;

  // Create an exact slice array
  std::vector<int64_t> slice_indices(v_local);
  for (int64_t i = 0; i < v_local; ++i)
    slice_indices[i] = v_start + i;
  Tensor slice_idx_tensor = Tensor::zeros(Shape{{v_local}}, idx_opts);
  cudaMemcpy(slice_idx_tensor.data<int64_t>(), slice_indices.data(),
             v_local * sizeof(int64_t), cudaMemcpyHostToDevice);

  // Initialize local portion
  Tensor full_weight_cpu = full_weight.to_cpu();
  float *fw_ptr = full_weight_cpu.data<float>();

  Tensor local_w_slice_cpu = Tensor::zeros(
      Shape{{v_local, C}},
      TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CPU));
  float *local_w_ptr = local_w_slice_cpu.data<float>();

  for (int64_t i = 0; i < v_local; ++i) {
    int64_t global_idx = v_start + i;
    for (int64_t j = 0; j < C; ++j) {
      local_w_ptr[i * C + j] = fw_ptr[global_idx * C + j];
    }
  }
  Tensor local_w_slice = local_w_slice_cpu.to_cuda(rank);
  dist_emb.weight->mutable_tensor().copy_(local_w_slice);
  if (dist_emb.weight->mutable_tensor().has_grad()) {
    dist_emb.weight->mutable_tensor().grad_view().fill(0.0f); // clear grad
  }

  Tensor dist_emb_out = dist_emb.forward(input_tensor);

  // Wrap in DTensor to add backward all-reduce hook.
  // The matmul backward produces dL/dx = dL/d_local_logits @ local_W,
  // which only has grad contributions from this rank's vocab shard.
  // The serial path sums across ALL V. We need to all-reduce grad_output
  // before it flows into VocabParallelEmbeddingBackward.
  Layout dt_emb_layout(mesh, {B, T, C});
  DTensor dt_emb_out(mesh, pg, dt_emb_layout, "dt_emb_out");
  dt_emb_out.mutable_tensor() = dist_emb_out;
  dt_emb_out.register_backward_node();

  // 2b. Mock Network (must match serial)
  Tensor dist_logits = dt_emb_out.mutable_tensor();

  // Reshape to 2D for matmul: [B*T, C]
  Tensor dist_logits_flat = autograd::reshape(dist_logits, Shape{{B * T, C}});

  // 2c. Distributed Loss
  // First, local logits
  Tensor local_head_weights =
      dist_emb.weight->mutable_tensor().t(); // [C, v_local]
  // [B*T, C] x [C, v_local] -> [B*T, v_local]
  Tensor dist_local_logits_flat =
      OwnTensor::autograd::matmul(dist_logits_flat, local_head_weights);

  // Reshape back to [B, T, v_local] for layout matching
  Tensor dist_local_logits =
      autograd::reshape(dist_local_logits_flat, Shape{{B, T, v_local}});

  Layout dp_layout(mesh, {B, T, V}, 2); // Sharded on V
  DTensor dt_logits(mesh, pg, dp_layout, "dt_logits");
  dt_logits.mutable_tensor() = dist_local_logits;

  Tensor dist_loss = vocab_parallel_cross_entropy_v2(dt_logits, target_tensor);

  // Backward
  dist_loss.backward();

  if (rank == 0) {
    std::cout << "\n[Dist] Loss: " << dist_loss.to_cpu().data<float>()[0]
              << std::endl;
  }

  // =========================================================================
  // PART 3: Verifications
  // =========================================================================

  bool passed = true;

  if (rank == 0)
    std::cout << "\n--- Verifications ---" << std::endl;

  // Check Forward Loss match
  if (rank == 0) {
    float sl = serial_loss.to_cpu().data<float>()[0];
    float dl = dist_loss.to_cpu().data<float>()[0];
    std::cout << "Loss Match: " << (std::abs(sl - dl) < 1e-4 ? "PASS" : "FAIL")
              << std::endl;
    if (std::abs(sl - dl) > 1e-4)
      passed = false;
  }

  // Check Backward Grad Match (Embedding Weight)
  // The serial grad is size [V, C]. The distributed grad is size [v_local, C].
  // Rank 0 should match rows 0..v_local. Rank 1 should match rows v_local..V.
  bool grad_match = true;
  if (dist_emb.weight->mutable_tensor().has_grad() && full_weight.has_grad()) {
    Tensor dist_gw = dist_emb.weight->mutable_tensor().grad_view().to_cpu();
    Tensor full_gw_cpu = full_weight.grad_view().to_cpu();

    float *d_ptr = dist_gw.data<float>();
    float *s_ptr = full_gw_cpu.data<float>();

    for (int64_t i = 0; i < v_local; ++i) {
      int64_t global_idx = v_start + i;
      for (int64_t j = 0; j < C; ++j) {
        float d_val = d_ptr[i * C + j];
        float s_val = s_ptr[global_idx * C + j];
        if (std::abs(d_val - s_val) > 1e-4) {
          grad_match = false;
          std::cout << "Rank " << rank << " Mismatch at Row " << i << " Col "
                    << j << " (Global " << global_idx << "): Dist=" << d_val
                    << ", Serial=" << s_val << std::endl;
          break;
        }
      }
      if (!grad_match)
        break;
    }

    std::cout << "Rank " << rank << " Dist Grad First Row: ";
    for (int j = 0; j < C; ++j)
      std::cout << d_ptr[j] << " ";
    std::cout << "\n";
    std::cout << "Rank " << rank << " Serial Grad Row " << v_start << ": ";
    for (int j = 0; j < C; ++j)
      std::cout << s_ptr[v_start * C + j] << " ";
    std::cout << "\n";
    // Print ALL dist grad rows vs serial for this rank
    for (int64_t i = 0; i < v_local; ++i) {
      int64_t gi = v_start + i;
      std::cout << "Rank " << rank << " Row " << i << " (Global " << gi
                << "): D=";
      std::cout << d_ptr[i * C] << " S=" << s_ptr[gi * C] << "\n";
    }
    std::cout << "Rank " << rank
              << " Embedding Grad Match: " << (grad_match ? "PASS" : "FAIL")
              << std::endl;
  } else {
    std::cout << "Rank " << rank << " Embedding Grad Match: MISSING GRADIENT"
              << std::endl;
    grad_match = false;
  }
  if (!grad_match)
    passed = false;

  if (rank == 0) {
    std::cout << "\nOVERALL TEST STATUS: "
              << (passed ? "SUCCESS - TENSOR PARALLEL MATH IS PERFECT"
                         : "FAILED")
              << std::endl;
  }

  MPI_Finalize();
  return 0;
}
