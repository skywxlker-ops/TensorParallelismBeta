#pragma once

#include "autograd/AutogradOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "core/Tensor.h"
#include "dnn/DistributedNN.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/dtensor.h"

#include "gpt2_cp_test/context_parallel/ContextParallelBackward.h"
#include "gpt2_cp_test/context_parallel/FusedSDPAOp.h"
#include "gpt2_cp_test/context_parallel/RingRotator.h"
#include "gpt2_cp_test/context_parallel/SDPAMerger.h"
#include "gpt2_cp_test/context_parallel/SDPAOp.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvtx3/nvToolsExt.h>

using namespace OwnTensor;
using namespace OwnTensor::dnn;

// ---------------------------------------------------------------------------
// RotatorType
//
// Selects which ring communication strategy to use.
// ---------------------------------------------------------------------------
enum class RotatorType {
  P2P,      // Point-to-point ncclSend/ncclRecv
  AlltoAll, // sendrecv-based ring shift
  AllGather // Single all_gather, then index
};

// ---------------------------------------------------------------------------
// CausalBehavior
//
// Controls how causal masking is applied during ring attention.
//
// FULL_CAUSAL:  Apply causal mask on every ring step (correct for
//               chunks where q_idx >= k_idx in the global sequence).
// SKIP_FUTURE:  Skip SDPA entirely for ring steps where the K,V chunk
//               is entirely in the "future" relative to the Q chunk.
// NOT_CAUSAL:   No causal masking (bidirectional attention).
// ---------------------------------------------------------------------------
enum class CausalBehavior { FULL_CAUSAL, SKIP_FUTURE, NOT_CAUSAL };

// ---------------------------------------------------------------------------
// ContextParallel : DModule
//
// Implements context parallelism (ring attention) as a distributed module.
//
// Forward:
//   1. Shard the input along the sequence dimension across ranks
//      (with optional load balancing via HeadTail permutation)
//   2. Each rank holds Q for its local chunk; K,V rotate around the ring
//   3. At each ring step: compute local SDPA, merge results via SDPAMerger
//   4. After all steps: unshard (gather) the output back to full sequence
//
// The module does NOT own any parameters -- it wraps the attention
// computation pattern for distributed execution.
// ---------------------------------------------------------------------------
class ContextParallel : public DModule {
public:
  ContextParallel(const DeviceMesh &mesh, std::shared_ptr<ProcessGroupNCCL> pg,
                  float attn_scale, bool is_causal = true,
                  RotatorType rotator_type = RotatorType::P2P,
                  bool load_balance = true, bool recompute_k = false)
      : mesh_(&mesh), pg_(pg), attn_scale_(attn_scale), is_causal_(is_causal),
        rotator_type_(rotator_type), load_balance_(load_balance),
        recompute_k_(recompute_k),
        world_size_(pg->get_worldsize()), rank_(pg->get_rank()) {}

  // -----------------------------------------------------------------------
  // forward
  //
  // Input: DTensor containing the full Q, K, V concatenated or separate.
  //        For this implementation, we expect q, k, v as separate Tensors
  //        already shaped as [B, H, T, D] (4D).
  //
  // This overload takes raw Tensors and returns the merged attention output.
  // -----------------------------------------------------------------------
  // unshard: when true (default), all-gather output to full [B,H,T,D].
  //          when false, output stays [B,H,T/n,D]; downstream layers work on
  //          the local sequence chunk with no inter-rank communication.
  // pre_sharded: when true, q/k/v are already [B,H,T/n,D] (e.g. from a
  //              previous layer that returned unshard=false). Phase 1 sharding
  //              is skipped. Must be paired with unshard=false.
  Tensor forward_cp(Tensor &q, // [B, H, T, D] -- full sequence query
                    Tensor &k, // [B, H, T, D] -- full sequence key
                    Tensor &v, // [B, H, T, D] -- full sequence value
                    bool unshard = true,
                    bool pre_sharded = false)
  {
    // ----- Phase 1: Context Parallel Shard -----
    // Shard Q along dim=2 (sequence dim in 4D [B, H, T, D])
    // K, V also sharded along dim=2

    // Load balance disabled for causal attention until LB backward grad
    // explosion is resolved. Non-causal uses LB when load_balance_=true.
    bool lb_active = load_balance_ && !is_causal_;

    // Input Q, K, V may be non-contiguous (e.g. from autograd::transpose
    // which swaps strides without copying data). Make contiguous BEFORE
    // sharding so that make_shards_inplace_axis produces views with
    // standard decreasing strides. Without this, the post-shard
    // contiguous() call produces wrong data because it cannot handle
    // non-standard stride ordering (stride[1] < stride[2]).
    Tensor q_work = autograd::contiguous(q);
    Tensor k_work = autograd::contiguous(k);
    Tensor v_work = autograd::contiguous(v);

    if (lb_active) {
      load_balancer_.set_world_size(world_size_);
      load_balancer_.set_chunk_dim(2); // sequence dim in 4D
      load_balancer_.set_stream(0);
      load_balancer_.loadbalance(q_work);
      load_balancer_.loadbalance(k_work);
      load_balancer_.loadbalance(v_work);
    }

    // ----- Phase 1: Shard (skipped when pre_sharded=true) -----
    // When pre_sharded=true, q/k/v are already [B,H,T/n,D] local chunks.
    Tensor local_q, local_k, local_v;
    if (pre_sharded) {
      // q_work is already autograd::contiguous(q) — local_q IS q_work
      local_q = q_work;
      local_k = k_work;
      local_v = v_work;
    } else {
      std::vector<Tensor> q_chunks =
          q_work.make_shards_inplace_axis(static_cast<size_t>(world_size_), 2);
      std::vector<Tensor> k_chunks =
          k_work.make_shards_inplace_axis(static_cast<size_t>(world_size_), 2);
      std::vector<Tensor> v_chunks =
          v_work.make_shards_inplace_axis(static_cast<size_t>(world_size_), 2);

      local_q = autograd::contiguous(q_chunks[rank_]); // [B, H, T/n, D]
      local_k = autograd::contiguous(k_chunks[rank_]); // [B, H, T/n, D]
      local_v = autograd::contiguous(v_chunks[rank_]); // [B, H, T/n, D]
    }

    // ----- Phase 2: Ring Attention Loop -----
    // Create rotator for K,V communication
    std::unique_ptr<RingRotatorBase> kv_rotator = create_rotator();

    // Initialize merger for accumulating partial attention outputs
    SDPAMerger merger(/*convert_to_f32=*/true);

    // Save K,V chunks, causal flags, partial flags, and per-step LSE for backward
    std::vector<Tensor> saved_k_chunks(world_size_);
    std::vector<Tensor> saved_v_chunks(world_size_);
    std::vector<bool> saved_causal_flags(world_size_, false);
    std::vector<bool> saved_partial_flags(world_size_, false);
    std::vector<Tensor> saved_lse_per_step(world_size_);

    // Sequence length of each rank's local chunk
    int64_t T_local_fwd = local_q.shape().dims[2];
    const int seq_dim = 2; // [B, H, T, D]

    // Pre-allocate KV send buffer (reused across ring steps)
    int64_t k_numel = local_k.numel();
    int64_t kv_numel = k_numel * 2;
    Tensor kv_send_buf = Tensor::empty(Shape({{kv_numel}}), local_k.opts());

    // Current K, V being processed (starts with local chunk)
    Tensor curr_k = local_k;
    Tensor curr_v = local_v;

    for (int i = 0; i < world_size_; ++i) {
      // Step 1: If not first iteration, get K,V from previous exchange
      if (i > 0) {
        Tensor next_kv = kv_rotator->next_buffer();
        Tensor kv_flat = next_kv.flatten();
        curr_k = kv_flat.narrow(0, 0, k_numel).reshape(local_k.shape());
        curr_v = kv_flat.narrow(0, k_numel, k_numel).reshape(local_v.shape());
      }

      // Step 2: Send current K,V to next rank (async, overlaps with compute)
      if (i < (world_size_ - 1)) {
        size_t k_bytes = static_cast<size_t>(k_numel) * sizeof(float);
        cudaMemcpyAsync(kv_send_buf.data<float>(), curr_k.data<float>(),
                        k_bytes, cudaMemcpyDeviceToDevice, 0);
        cudaMemcpyAsync(kv_send_buf.data<float>() + k_numel,
                        curr_v.data<float>(), k_bytes, cudaMemcpyDeviceToDevice,
                        0);
        nvtxRangePushA("CP.fwd.ring.exchange_buffers");
        kv_rotator->exchange_buffers(kv_send_buf);
        nvtxRangePop();
      }

      // Step 3: Determine causal behavior for this ring step
      // With load balance: i==0 is causal, all others are NOT_CAUSAL (never SKIP)
      // Without load balance: i==0 is causal, past chunks full attention, future skipped
      bool skip_step = false;
      bool use_causal = false;
      if (is_causal_) {
        if (i == 0) {
          use_causal = true;
        } else if (lb_active) {
          // Load balance enabled: never skip, never causal on non-diagonal
          use_causal = false;
        } else {
          // No load balance: skip future chunks
          int source_rank =
              ((rank_ - i) % world_size_ + world_size_) % world_size_;
          if (source_rank > rank_) {
            skip_step = true;
          }
          use_causal = false;
        }
      }

      if (skip_step) {
        continue;
      }

      // Step 4: Sub-chunk Q, K, V for load-balanced dispatch
      // Matches PyTorch _templated_ring_attention lines 451-471:
      //   i==0:       full Q, K, V (causal)
      //   i<=rank LB: full Q, K[:T/2], V[:T/2] (not causal, partial=false)
      //   i>rank  LB: Q[T/2:], full K, V (not causal, partial=true)
      Tensor q_use = local_q;
      Tensor k_use = curr_k;
      Tensor v_use = curr_v;
      bool use_partial = false;

      if (lb_active && i > 0) {
        if (i <= rank_) {
          // Past chunk with LB: full Q, first half of K/V
          std::vector<Tensor> k_halves =
              curr_k.make_shards_inplace_axis(2, seq_dim);
          std::vector<Tensor> v_halves =
              curr_v.make_shards_inplace_axis(2, seq_dim);
          k_use = autograd::contiguous(k_halves[0]);
          v_use = autograd::contiguous(v_halves[0]);
        } else {
          // Future chunk with LB: second half of Q, full K/V
          std::vector<Tensor> q_halves =
              local_q.make_shards_inplace_axis(2, seq_dim);
          q_use = autograd::contiguous(q_halves[1]);
          use_partial = true;
        }
      }

      // Save full (un-chunked) K,V for backward before computing.
      // When recompute_k_=true, only save step 0 (local K,V) as the starting
      // point for backward re-rotation. Other steps are recomputed.
      if (!recompute_k_ || i == 0) {
        saved_k_chunks[i] = curr_k.clone();
        saved_v_chunks[i] = curr_v.clone();
      }
      saved_causal_flags[i] = use_causal;
      saved_partial_flags[i] = use_partial;

      // Step 5: Compute fused SDPA
      // With LB sub-chunking: no cross-chunk causal offsets needed (is_causal
      // only on diagonal, sub-chunks handle the rest).
      // Without LB: offsets needed for cross-chunk causal masking.
      int q_off = 0;
      int k_off = 0;
      if (!lb_active && is_causal_) {
        int source_rank =
            ((rank_ - i) % world_size_ + world_size_) % world_size_;
        q_off = rank_ * static_cast<int>(T_local_fwd);
        k_off = source_rank * static_cast<int>(T_local_fwd);
      }
      SDPAResult result = sdpa_fused_forward(
          q_use, k_use, v_use, use_causal, attn_scale_, q_off, k_off);

      // Save per-step LSE for backward
      saved_lse_per_step[i] = result.lse;

      // Step 6: Merge into accumulator (with partial flag)
      merger.step(result.out, result.lse, use_partial);
    }

    // ----- Phase 3: Get final merged result -----
    auto [merged_out, merged_lse] = merger.results();

    // ----- Phase 4: Context Parallel Unshard -----
    // When unshard=false: skip allgather; return merged_out [B,H,T/n,D].
    // Downstream layers (MLP, LayerNorm, loss) work on the local T/n chunk.
    int64_t B = local_q.shape().dims[0];
    int64_t H = local_q.shape().dims[1];
    int64_t T_local = local_q.shape().dims[2];
    int64_t D = local_q.shape().dims[3];

    Tensor output_tensor;
    if (!unshard) {
      // No allgather: output stays [B, H, T/n, D]
      output_tensor = merged_out;
    } else {
      // Gather the output chunks back to full [B, H, T, D]
      size_t local_count = static_cast<size_t>(merged_out.numel());
      size_t total_count = local_count * static_cast<size_t>(world_size_);

      Shape gathered_shape({{static_cast<int64_t>(total_count)}});
      Tensor gathered_flat = Tensor::empty(gathered_shape, merged_out.opts());

      nvtxRangePushA("CP.fwd.unshard.all_gather");
      pg_->all_gather(merged_out.data<float>(), gathered_flat.data<float>(),
                      local_count, merged_out.dtype(),
                      true); // sync
      nvtxRangePop();

      int64_t T_full = T_local * world_size_;
      Shape full_shape({{B, H, T_full, D}});
      Tensor full_output = Tensor::empty(full_shape, merged_out.opts());

      size_t slice_bytes = static_cast<size_t>(T_local * D) * sizeof(float);
      for (int r = 0; r < world_size_; ++r) {
        for (int64_t b = 0; b < B; ++b) {
          for (int64_t h = 0; h < H; ++h) {
            float *src =
                gathered_flat.data<float>() + r * (B * H * T_local * D) +
                b * (H * T_local * D) + h * (T_local * D);
            float *dst = full_output.data<float>() + b * (H * T_full * D) +
                         h * (T_full * D) + r * (T_local * D);
            cudaMemcpyAsync(dst, src, slice_bytes, cudaMemcpyDeviceToDevice, 0);
          }
        }
      }
      cudaStreamSynchronize(0);

      if (lb_active) {
        load_balancer_.unloadbalance(full_output);
      }
      output_tensor = full_output;
    }

    // ----- Register backward node -----
    if (q.requires_grad() || k.requires_grad() || v.requires_grad()) {
      int rot_type = static_cast<int>(rotator_type_);
      // Pass merged_out.detach() to break the cycle:
      // output_tensor -> grad_fn -> merged_out -> output_tensor (when unshard=false
      // output_tensor IS merged_out). Detach severs the grad_fn strong-ptr path.
      auto grad_fn = std::make_shared<ContextParallelBackward>(
          local_q, saved_k_chunks, saved_v_chunks, saved_causal_flags,
          saved_partial_flags, saved_lse_per_step, merged_lse,
          merged_out.detach(), pg_, attn_scale_, is_causal_, rot_type,
          lb_active, world_size_, rank_, unshard, recompute_k_);

      if (q.requires_grad()) {
        Tensor &q_mut = const_cast<Tensor &>(q);
        grad_fn->set_next_edge(0, autograd::get_grad_edge(q_mut));
      }
      if (k.requires_grad()) {
        Tensor &k_mut = const_cast<Tensor &>(k);
        grad_fn->set_next_edge(1, autograd::get_grad_edge(k_mut));
      }
      if (v.requires_grad()) {
        Tensor &v_mut = const_cast<Tensor &>(v);
        grad_fn->set_next_edge(2, autograd::get_grad_edge(v_mut));
      }

      output_tensor.set_grad_fn(grad_fn);
      output_tensor.set_requires_grad(true);
    }

    return output_tensor;
  }

private:
  const DeviceMesh *mesh_;
  std::shared_ptr<ProcessGroupNCCL> pg_;
  float attn_scale_;
  bool is_causal_;
  RotatorType rotator_type_;
  bool load_balance_;
  bool recompute_k_;
  int world_size_;
  int rank_;
  HeadTail load_balancer_;

  std::unique_ptr<RingRotatorBase> create_rotator() const {
    switch (rotator_type_) {
    case RotatorType::P2P:
      return std::make_unique<P2PRingRotator>(pg_);
    case RotatorType::AlltoAll:
      return std::make_unique<AlltoAllRingRotator>(pg_);
    case RotatorType::AllGather:
      return std::make_unique<AllGatherRingRotator>(pg_);
    default:
      throw std::runtime_error("Unknown rotator type");
    }
  }
};
