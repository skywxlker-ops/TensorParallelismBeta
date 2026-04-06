#include "dnn/DistributedNN.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace OwnTensor;
using namespace OwnTensor::dnn;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  cudaSetDevice(rank);

  std::vector<int> ranks_vec;
  for (int i = 0; i < world_size; i++)
    ranks_vec.push_back(i);
  DeviceMesh mesh({world_size}, ranks_vec);
  auto pg = mesh.get_process_group(0);

  int64_t B = 2;
  int64_t T = 4;
  int64_t C = 8;
  int64_t V = 16;

  TensorOptions idx_opts =
      TensorOptions().with_dtype(Dtype::Int64).with_device(Device::CPU);
  TensorOptions float_opts =
      TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CPU);

  // 1. Input Tokens
  Tensor input_cpu = Tensor::zeros(Shape{{B, T}}, idx_opts);
  int64_t *in_ptr = input_cpu.data<int64_t>();
  for (int i = 0; i < B * T; ++i)
    in_ptr[i] = i % V;
  Tensor input_tensor = input_cpu.to_cuda(rank);

  // 2. Targets
  Tensor target_cpu = Tensor::zeros(Shape{{B, T}}, idx_opts);
  int64_t *tgt_ptr = target_cpu.data<int64_t>();
  for (int i = 0; i < B * T; ++i)
    tgt_ptr[i] = (i + 1) % V;
  Tensor target_tensor = target_cpu.to_cuda(rank);

  // 3. Shared Weight
  Tensor full_weight_cpu = Tensor::zeros(Shape{{V, C}}, float_opts);
  float *fw_ptr = full_weight_cpu.data<float>();
  for (int i = 0; i < V * C; ++i)
    fw_ptr[i] = (i % 100) * 0.01f - 0.5f;
  Tensor full_weight = full_weight_cpu.to_cuda(rank);
  full_weight.set_requires_grad(true);

  // =========================================================================
  // PART 1: SERIAL IMPLEMENTATION
  // =========================================================================
  Tensor serial_emb_out = autograd::embedding(full_weight, input_tensor);
  Tensor serial_logits = autograd::matmul(serial_emb_out, full_weight.t());

  Tensor serial_logits_flat =
      autograd::reshape(serial_logits, Shape{{B * T, V}});
  Tensor serial_targets_flat = autograd::reshape(target_tensor, Shape{{B * T}});
  Tensor serial_loss = autograd::sparse_cross_entropy_loss(serial_logits_flat,
                                                           serial_targets_flat);

  serial_loss.backward();

  // =========================================================================
  // PART 2: DISTRIBUTED IMPLEMENTATION WITH TIED WEIGHTS
  // =========================================================================
  DEmbeddingVParallelFused dist_emb(mesh, pg, V, C, 0.02f, 42);

  int64_t v_local = V / world_size;
  int64_t v_start = rank * v_local;

  Tensor local_w_slice_cpu = Tensor::zeros(Shape{{v_local, C}}, float_opts);
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
    dist_emb.weight->mutable_tensor().grad_view().fill(0.0f);
  }

  DLMHead lm_head(mesh, pg, B, T, C, V, true, dist_emb.weight.get());

  Tensor dist_emb_out = dist_emb.forward(input_tensor);

  Layout dt_emb_layout(mesh, {B, T, C});
  DTensor dt_emb_dt(mesh, pg, dt_emb_layout, "dt_emb_dt");
  dt_emb_dt.mutable_tensor() = dist_emb_out;
  dt_emb_dt.register_backward_node();

  DTensor dt_logits = lm_head.forward(dt_emb_dt);
  Tensor dist_loss = vocab_parallel_cross_entropy_v2(dt_logits, target_tensor);

  dist_loss.backward();

  if (rank == 0) {
    std::cout << "\n[Dist] Loss: " << dist_loss.to_cpu().data<float>()[0]
              << std::endl;
    std::cout << "[Serial] Loss: " << serial_loss.to_cpu().data<float>()[0]
              << std::endl;
    std::cout << "Loss Match: "
              << (std::abs(dist_loss.to_cpu().data<float>()[0] -
                           serial_loss.to_cpu().data<float>()[0]) < 1e-4
                      ? "PASS"
                      : "FAIL")
              << std::endl;
  }

  Tensor dist_grad = dist_emb.weight->mutable_tensor().grad_view().to_cpu();
  Tensor serial_grad = full_weight.grad_view().to_cpu();

  float *dg_ptr = dist_grad.data<float>();
  float *sg_ptr = serial_grad.data<float>();

  bool grads_match = true;
  for (int64_t i = 0; i < v_local; ++i) {
    int64_t global_idx = v_start + i;
    for (int64_t j = 0; j < C; ++j) {
      float dg = dg_ptr[i * C + j];
      float sg = sg_ptr[global_idx * C + j];
      if (std::abs(dg - sg) > 1e-4) {
        grads_match = false;
        std::cout << "Rank " << rank << " Mismatch at global " << global_idx
                  << " (" << i << "," << j << "): D=" << dg << " S=" << sg
                  << std::endl;
      }
    }
  }

  if (rank == 0) {
    std::cout << "Rank 0 Embedding Grad Match: "
              << (grads_match ? "PASS" : "FAIL") << std::endl;
  }
  if (rank == 1) {
    std::cout << "Rank 1 Embedding Grad Match: "
              << (grads_match ? "PASS" : "FAIL") << std::endl;
  }

  pg.reset();
  MPI_Finalize();
  return 0;
}
