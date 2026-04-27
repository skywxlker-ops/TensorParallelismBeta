#pragma once

#include "autograd/Engine.h"
#include "autograd/Node.h"
#include "autograd/ops_template.h"
#include "core/Tensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/dtensor.h"

#include "gpt2_cp_test/context_parallel/FusedSDPAOp.h"
#include "gpt2_cp_test/context_parallel/RingRotator.h"
#include "gpt2_cp_test/context_parallel/SDPAMerger.h"
#include "gpt2_cp_test/context_parallel/SDPAOp.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nvtx3/nvToolsExt.h>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// ContextParallelBackward
//
// Autograd node for the backward pass of context parallel ring attention.
//
// Key corrections vs naive backward:
//   1. Shards the incoming full gradient [B,H,T,D] to local [B,H,T/n,D]
//   2. Applies merger rescaling: each step's grad is weighted by
//      exp(step_lse - merged_lse) to account for the online softmax merge
//   3. Communicates dK/dV back to source ranks via sendrecv
//   4. All-gathers local gradients to reconstruct full [B,H,T,D] gradients
// ---------------------------------------------------------------------------
class ContextParallelBackward : public Node {
public:
  ContextParallelBackward(
      // Saved tensors from forward
      Tensor saved_q,                         // [B, H, T/n, D]
      std::vector<Tensor> saved_k_chunks,     // K chunks per ring step
      std::vector<Tensor> saved_v_chunks,     // V chunks per ring step
      std::vector<bool> saved_causal_flags,   // causal flag per ring step
      std::vector<bool> saved_partial_flags,  // partial flag per ring step
      std::vector<Tensor> saved_lse_per_step, // LSE per ring step [B,H,T/n,1]
      Tensor merged_lse,                      // final merged LSE [B,H,T/n,1]
      Tensor merged_out,                      // final merged out [B,H,T/n,D]
      // Process group and config
      std::shared_ptr<ProcessGroupNCCL> pg, float attn_scale, bool is_causal,
      int rotator_type, bool load_balance, int world_size,
      int rank, bool unshard = true,
      bool recompute_k = true)
      : Node(3), // 3 outputs: grad for q, k, v
        saved_q_(saved_q), saved_k_chunks_(saved_k_chunks),
        saved_v_chunks_(saved_v_chunks),
        saved_causal_flags_(saved_causal_flags),
        saved_partial_flags_(saved_partial_flags),
        saved_lse_per_step_(saved_lse_per_step), merged_lse_(merged_lse),
        merged_out_(merged_out), pg_(pg), attn_scale_(attn_scale),
        is_causal_(is_causal), rotator_type_(rotator_type),
        load_balance_(load_balance), world_size_(world_size), rank_(rank),
        unshard_(unshard), recompute_k_(recompute_k) {}

  const char *name() const override { return "ContextParallelBackward"; }

  std::vector<Tensor> apply(std::vector<Tensor> &&grads) override {
    if (grads.empty()) {
      throw std::runtime_error(
          "ContextParallelBackward: no gradients provided");
    }

    // ----- Phase 0: Shard the full gradient to local chunk -----
    Tensor grad_local;
    if (unshard_) {
      Tensor grad_output_full = grads[0].contiguous();
      std::vector<Tensor> grad_chunks =
          grad_output_full.make_shards_inplace_axis(
              static_cast<size_t>(world_size_), 2);
      grad_local = grad_chunks[rank_].contiguous();
    } else {
      grad_local = grads[0].contiguous();
    }

    // ----- Phase 1: Gradient Buffer Init -----
    // Single travelling accumulators for dK/dV (rotated through ring).
    // This matches PyTorch's pipelined dkv_rotater protocol.
    const int seq_dim = 2;
    const int64_t T_local_bwd = saved_q_.shape().dims[seq_dim];

    Tensor grad_q = Tensor::zeros(saved_q_.shape(), saved_q_.opts());
    Tensor grad_key = Tensor::zeros(
        saved_k_chunks_[0].shape(), saved_k_chunks_[0].opts());
    Tensor grad_value = Tensor::zeros(
        saved_v_chunks_[0].shape(), saved_v_chunks_[0].opts());

    // dkv_rotater: pipelined rotation of dK/dV (only for LB path)
    std::unique_ptr<RingRotatorBase> dkv_rotater;
    // Batch accumulators for non-LB path (old Phase 3 approach)
    std::vector<Tensor> grad_k_accum;
    std::vector<Tensor> grad_v_accum;

    bool lb_active_bwd = load_balance_;
    if (lb_active_bwd) {
      dkv_rotater = create_rotator();
    } else {
      grad_k_accum.resize(world_size_);
      grad_v_accum.resize(world_size_);
    }

    // Optional kv_rotater for recompute_k mode
    std::unique_ptr<RingRotatorBase> kv_rotater;
    Tensor curr_k, curr_v;
    if (recompute_k_) {
      kv_rotater = create_rotator();
      curr_k = saved_k_chunks_[0];
      curr_v = saved_v_chunks_[0];
    }

    // ----- Phase 2: Ring Loop (backward) -----
    for (int i = 0; i < world_size_; ++i) {
      // --- K/V access ---
      Tensor step_k, step_v;
      if (recompute_k_) {
        // Recompute: rotate K/V through ring (same protocol as forward)
        if (i > 0) {
          Tensor next_kv = kv_rotater->next_buffer();
          Tensor kv_flat = next_kv.flatten();
          int64_t k_numel = curr_k.numel();
          curr_k = kv_flat.narrow(0, 0, k_numel).reshape(curr_k.shape());
          curr_v = kv_flat.narrow(0, k_numel, k_numel).reshape(curr_v.shape());
        }
        if (i < (world_size_ - 1)) {
          int64_t k_numel = curr_k.numel();
          size_t k_bytes = static_cast<size_t>(k_numel) * sizeof(float);
          Tensor kv_send =
              Tensor::empty(Shape({{k_numel * 2}}), curr_k.opts());
          cudaMemcpyAsync(kv_send.data<float>(), curr_k.data<float>(),
                          k_bytes, cudaMemcpyDeviceToDevice, 0);
          cudaMemcpyAsync(kv_send.data<float>() + k_numel,
                          curr_v.data<float>(), k_bytes,
                          cudaMemcpyDeviceToDevice, 0);
          kv_rotater->exchange_buffers(kv_send);
        }
        step_k = curr_k;
        step_v = curr_v;
      } else {
        // Save-K path (default): use pre-saved K/V
        // NOTE: Do NOT continue here -- dkv_rotater communication below must
        // happen on every step to avoid NCCL deadlocks across ranks.
        step_k = saved_k_chunks_[i];
        step_v = saved_v_chunks_[i];
      }

      // If this step was skipped in forward (invalid K/V), skip SDPA but
      // still participate in dkv_rotater communication below.
      bool step_skipped = !step_k.is_valid();

      bool use_causal = saved_causal_flags_[i];
      bool use_partial = saved_partial_flags_[i];

      // --- Compute SDPA backward (skip if step was skipped in forward) ---
      Tensor grad_q_step, grad_k_step, grad_v_step;

      if (!step_skipped) {
        Tensor q_bwd = saved_q_;
        Tensor k_bwd = step_k;
        Tensor v_bwd = step_v;
        Tensor out_bwd = merged_out_;
        Tensor grad_out_bwd = grad_local;
        Tensor lse_bwd = merged_lse_;

        if (lb_active_bwd && i > 0) {
          if (i <= rank_) {
            // Past with LB: full Q, first half K/V, full out/grad_out/lse
            std::vector<Tensor> k_halves =
                step_k.make_shards_inplace_axis(2, seq_dim);
            std::vector<Tensor> v_halves =
                step_v.make_shards_inplace_axis(2, seq_dim);
            k_bwd = k_halves[0].contiguous();
            v_bwd = v_halves[0].contiguous();
          } else {
            // Future with LB: 2nd half of Q, out, grad_out, lse; full K/V
            std::vector<Tensor> q_halves =
                saved_q_.make_shards_inplace_axis(2, seq_dim);
            q_bwd = q_halves[1].contiguous();

            std::vector<Tensor> out_halves =
                merged_out_.make_shards_inplace_axis(2, seq_dim);
            out_bwd = out_halves[1].contiguous();

            std::vector<Tensor> grad_halves =
                grad_local.make_shards_inplace_axis(2, seq_dim);
            grad_out_bwd = grad_halves[1].contiguous();

            std::vector<Tensor> lse_halves =
                merged_lse_.make_shards_inplace_axis(2, seq_dim);
            lse_bwd = lse_halves[1].contiguous();
          }
        }

        int q_off = 0, k_off = 0;
        if (!lb_active_bwd && is_causal_) {
          int source_rank =
              ((rank_ - i) % world_size_ + world_size_) % world_size_;
          q_off = rank_ * static_cast<int>(T_local_bwd);
          k_off = source_rank * static_cast<int>(T_local_bwd);
        }
        std::vector<Tensor> step_grads = sdpa_fused_backward(
            q_bwd, k_bwd, v_bwd, grad_out_bwd, out_bwd, lse_bwd,
            use_causal, attn_scale_, q_off, k_off);

        grad_q_step = step_grads[0];
        grad_k_step = step_grads[1];
        grad_v_step = step_grads[2];

        // --- Accumulate dQ (local, never rotated) ---
        if (lb_active_bwd && i > rank_) {
          int64_t half_T = T_local_bwd / 2;
          Tensor gq_1st = grad_q.narrow_view(seq_dim, 0, half_T);
          Tensor gq_2nd = grad_q.narrow_view(seq_dim, half_T, half_T);
          grad_q = Tensor::cat({gq_1st.clone(), gq_2nd.clone() + grad_q_step}, seq_dim);
        } else if (grad_q_step.is_valid()) {
          grad_q = grad_q + grad_q_step;
        }
      }

      // --- dK/dV accumulation ---
      if (lb_active_bwd) {
        // LB path: pipelined rotation via dkv_rotater (PyTorch lines 588-627)
        if (i == 0) {
          if (grad_k_step.is_valid()) {
            grad_key = grad_key + grad_k_step;
          }
          if (grad_v_step.is_valid()) {
            grad_value = grad_value + grad_v_step;
          }
        } else {
          int64_t k_numel = grad_key.numel();
          Tensor next_grad_kv = dkv_rotater->next_buffer();
          Tensor gkv_flat = next_grad_kv.flatten();
          grad_key =
              gkv_flat.narrow(0, 0, k_numel).reshape(grad_key.shape());
          grad_value =
              gkv_flat.narrow(0, k_numel, k_numel).reshape(grad_value.shape());

          if (i <= rank_ && grad_k_step.is_valid()) {
            int64_t half_T = T_local_bwd / 2;
            Tensor zeros_half_k = Tensor::zeros(grad_k_step.shape(), grad_k_step.opts());
            Tensor gk_padded = Tensor::cat({grad_k_step, zeros_half_k}, seq_dim);
            Tensor zeros_half_v = Tensor::zeros(grad_v_step.shape(), grad_v_step.opts());
            Tensor gv_padded = Tensor::cat({grad_v_step, zeros_half_v}, seq_dim);
            grad_key = grad_key + gk_padded;
            grad_value = grad_value + gv_padded;
          } else {
            if (grad_k_step.is_valid()) {
              grad_key = grad_key + grad_k_step;
            }
            if (grad_v_step.is_valid()) {
              grad_value = grad_value + grad_v_step;
            }
          }
        }

        // Send current grad_key/grad_value to next rank
        int64_t k_numel_send = grad_key.numel();
        size_t k_bytes_send =
            static_cast<size_t>(k_numel_send) * sizeof(float);
        Tensor grad_kv_send =
            Tensor::empty(Shape({{k_numel_send * 2}}), grad_key.opts());
        cudaMemcpyAsync(grad_kv_send.data<float>(), grad_key.data<float>(),
                        k_bytes_send, cudaMemcpyDeviceToDevice, 0);
        cudaMemcpyAsync(grad_kv_send.data<float>() + k_numel_send,
                        grad_value.data<float>(), k_bytes_send,
                        cudaMemcpyDeviceToDevice, 0);
        nvtxRangePushA("CP.bwd.LB.ring.exchange_buffers");
        dkv_rotater->exchange_buffers(grad_kv_send);
        nvtxRangePop();
      } else {
        // Non-LB path: batch accumulate per step, single sendrecv at end
        if (!step_skipped && grad_k_step.is_valid()) {
          grad_k_accum[i] = grad_k_step;
          grad_v_accum[i] = grad_v_step;
        }
      }
    }

    // --- Final: collect dK/dV from ring ---
    if (lb_active_bwd) {
      // LB: receive completed travelling accumulator
      int64_t k_numel = grad_key.numel();
      nvtxRangePushA("CP.bwd.LB.post_loop.next_buffer");
      Tensor final_grad_kv = dkv_rotater->next_buffer();
      nvtxRangePop();
      Tensor final_flat = final_grad_kv.flatten();
      grad_key =
          final_flat.narrow(0, 0, k_numel).reshape(grad_key.shape());
      grad_value =
          final_flat.narrow(0, k_numel, k_numel).reshape(grad_value.shape());
    } else {
      // Non-LB: step 0 is local, steps 1..N-1 exchanged via packed sendrecv.
      // grad_key/grad_value (initialized to zeros at lines 88-92) accumulate results.

      // Step 0: local contribution (our own K chunk)
      if (grad_k_accum[0].is_valid()) {
        grad_key = grad_key + grad_k_accum[0];
        grad_value = grad_value + grad_v_accum[0];
      }

      // Steps 1..N-1: pack K+V, single sendrecv per step
      for (int i = 1; i < world_size_; ++i) {
        int source_rank = ((rank_ - i) % world_size_ + world_size_) % world_size_;
        int dest_rank = ((rank_ + i) % world_size_ + world_size_) % world_size_;

        int64_t k_numel = saved_k_chunks_[0].numel();
        size_t k_bytes = static_cast<size_t>(k_numel) * sizeof(float);

        // Pack K+V into one send buffer
        Tensor send_buf = Tensor::zeros(Shape({{k_numel * 2}}),
                                        saved_k_chunks_[0].opts());
        if (grad_k_accum[i].is_valid()) {
          cudaMemcpyAsync(send_buf.data<float>(), grad_k_accum[i].data<float>(),
                          k_bytes, cudaMemcpyDeviceToDevice, 0);
          cudaMemcpyAsync(send_buf.data<float>() + k_numel,
                          grad_v_accum[i].data<float>(),
                          k_bytes, cudaMemcpyDeviceToDevice, 0);
        }

        Tensor recv_buf = Tensor::empty(Shape({{k_numel * 2}}),
                                        saved_k_chunks_[0].opts());

        // Single packed sendrecv
        nvtxRangePushA("CP.bwd.nonLB.sendrecv");
        pg_->sendrecv(send_buf.data<float>(), recv_buf.data<float>(),
                      source_rank, dest_rank,
                      static_cast<size_t>(k_numel * 2),
                      saved_k_chunks_[0].dtype(), true);
        nvtxRangePop();

        // Unpack and accumulate
        Tensor recv_k = recv_buf.narrow(0, 0, k_numel).reshape(
            saved_k_chunks_[0].shape());
        Tensor recv_v = recv_buf.narrow(0, k_numel, k_numel).reshape(
            saved_v_chunks_[0].shape());

        grad_key = grad_key + recv_k;
        grad_value = grad_value + recv_v;
      }
    }

    // Cast back to original dtype
    if (grad_q.dtype() != saved_q_.dtype()) {
      grad_q = grad_q.as_type(saved_q_.dtype());
    }
    if (grad_key.dtype() != saved_k_chunks_[0].dtype()) {
      grad_key = grad_key.as_type(saved_k_chunks_[0].dtype());
      grad_value = grad_value.as_type(saved_v_chunks_[0].dtype());
    }

    // ----- Phase 4: Unshard gradients -----
    if (!unshard_) {
      return {grad_q, grad_key, grad_value};
    }

    nvtxRangePushA("CP.bwd.unshard.all_gather.grad_q");
    Tensor full_grad_q = all_gather_along_seq(grad_q);
    nvtxRangePop();
    nvtxRangePushA("CP.bwd.unshard.all_gather.grad_k");
    Tensor full_grad_k = all_gather_along_seq(grad_key);
    nvtxRangePop();
    nvtxRangePushA("CP.bwd.unshard.all_gather.grad_v");
    Tensor full_grad_v = all_gather_along_seq(grad_value);
    nvtxRangePop();

    bool lb_active = load_balance_;
    if (lb_active) {
      HeadTail lb;
      lb.set_world_size(world_size_);
      lb.set_chunk_dim(2);
      lb.set_stream(0);
      lb.unloadbalance(full_grad_q);
      lb.unloadbalance(full_grad_k);
      lb.unloadbalance(full_grad_v);
    }

    return {full_grad_q, full_grad_k, full_grad_v};
  }

  void release_saved_variables() override {
    saved_q_ = Tensor();
    saved_k_chunks_.clear();
    saved_v_chunks_.clear();
    saved_lse_per_step_.clear();
    merged_lse_ = Tensor();
    merged_out_ = Tensor();
  }

private:
  Tensor saved_q_;
  std::vector<Tensor> saved_k_chunks_;
  std::vector<Tensor> saved_v_chunks_;
  std::vector<bool> saved_causal_flags_;
  std::vector<bool> saved_partial_flags_;
  std::vector<Tensor> saved_lse_per_step_;
  Tensor merged_lse_;
  Tensor merged_out_;

  std::shared_ptr<ProcessGroupNCCL> pg_;
  float attn_scale_;
  bool is_causal_;
  int rotator_type_;
  bool load_balance_;
  int world_size_;
  int rank_;
  bool unshard_;
  bool recompute_k_;

  std::unique_ptr<RingRotatorBase> create_rotator() const {
    switch (rotator_type_) {
    case 0:
      return std::make_unique<P2PRingRotator>(pg_);
    case 1:
      return std::make_unique<AlltoAllRingRotator>(pg_);
    case 2:
      return std::make_unique<AllGatherRingRotator>(pg_);
    default:
      throw std::runtime_error("Unknown rotator type");
    }
  }

  Tensor all_gather_along_seq(Tensor &local) {
    size_t local_count = static_cast<size_t>(local.numel());
    size_t total_count = local_count * static_cast<size_t>(world_size_);

    Shape flat_shape({{static_cast<int64_t>(total_count)}});
    Tensor gathered_flat = Tensor::empty(flat_shape, local.opts());

    pg_->all_gather(local.data<float>(), gathered_flat.data<float>(),
                    local_count, local.dtype(), true);

    int64_t B = local.shape().dims[0];
    int64_t H = local.shape().dims[1];
    int64_t T_local = local.shape().dims[2];
    int64_t D = local.shape().dims[3];
    int64_t T_full = T_local * world_size_;

    Shape full_shape({{B, H, T_full, D}});
    Tensor full = Tensor::empty(full_shape, local.opts());

    // Per-(b,h)-slice copy with correct strides (see ContextParallel.h)
    size_t slice_bytes = static_cast<size_t>(T_local * D) * sizeof(float);
    for (int r = 0; r < world_size_; ++r) {
      for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
          float *src = gathered_flat.data<float>() + r * (B * H * T_local * D) +
                       b * (H * T_local * D) + h * (T_local * D);
          float *dst = full.data<float>() + b * (H * T_full * D) +
                       h * (T_full * D) + r * (T_local * D);
          cudaMemcpyAsync(dst, src, slice_bytes, cudaMemcpyDeviceToDevice, 0);
        }
      }
    }
    cudaStreamSynchronize(0);

    return full;
  }
};
