#pragma once

#include "autograd/AutogradOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "core/Tensor.h"
#include "dnn/DistributedNN.h"
#include "ops/IndexingOps.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/dtensor.h"

#include "gpt2_cp_test/context_parallel/ContextParallelBackward.h"
#include "gpt2_cp_test/context_parallel/FusedSDPAOp.h"
#include "gpt2_cp_test/context_parallel/KVPackKernel.h"
#include "gpt2_cp_test/context_parallel/RingRotator.h"
#include "gpt2_cp_test/context_parallel/SDPAMerger.h"
#include "gpt2_cp_test/context_parallel/SDPAOp.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
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
// Pre-embedding sequence sharding helper.
//
// Mirrors PyTorch's context_parallel() buffer pre-sharding: takes full-T
// token ids (and optional targets) on the device, returns this rank's
// [B, T/n] slice + matching [1, T/n] position indices.
//
//   load_balance=false: contiguous chunk [rank*T_local, (rank+1)*T_local)
//   load_balance=true : HeadTail permutation (out[2k]=k, out[2k+1]=T-1-k),
//                       then take this rank's contiguous slice.
//
// Note: HeadTail mode produces a runnable pipeline (shapes + grads OK) but
// the loss is NOT numerically correct for causal attention until the ring
// loop's causal mask is upgraded to the 3-mask sub-chunk scheme. Contiguous
// mode is fully correct.
// ---------------------------------------------------------------------------
struct ShardedInputs {
  Tensor idx_local;  // [B, T/n] int64
  Tensor pos_local;  // [1, T/n] int64
  Tensor y_local;    // [B, T/n] int64 (invalid Tensor if y_full was invalid)
};

inline ShardedInputs shard_sequence_pre_embed(
    const Tensor &idx_full,
    const Tensor &y_full,
    int64_t T_full,
    int world_size,
    int rank,
    bool load_balance,
    DeviceIndex device) {
  if (T_full % world_size != 0) {
    throw std::invalid_argument(
        "shard_sequence_pre_embed: T_full must be divisible by world_size");
  }
  int64_t T_local = T_full / world_size;
  int64_t B = idx_full.shape().dims[0];

  // Build per-rank absolute position list on CPU.
  //
  // Two layouts:
  //   load_balance=false: contiguous chunk [rank*T_local, (rank+1)*T_local).
  //   load_balance=true : PyTorch chunk-level HeadTail. Sequence is split into
  //     2*world_size equal chunks of size chunk_sz = T_full/(2*N). Rank r owns
  //     chunks (r, 2N-1-r), concatenated as [head_chunk, tail_chunk]. This
  //     gives each rank a contiguous "early" half and a contiguous "late" half
  //     so the round-robin sub-chunk causal dispatch in forward_cp works
  //     correctly (first half = early global positions, second half = late).
  //     Matches PyTorch _attention.py @ e9ebbd3b _rearrange_seq_for_load_balance.
  //
  // TODO(headtail-consolidation): This CPU-side perm_local construction
  // duplicates the chunk-level math now implemented in
  // tensor/headtail_kernel.cu (HeadTail::loadbalance). Both MUST stay
  // in sync: chunk_sz = T/(2*N), rank r owns chunks (r, 2N-1-r)
  // concatenated as [head_chunk, tail_chunk]. A future cleanup should
  // replace this block with a call to HeadTail::loadbalance followed by
  // make_shards_inplace_axis. Until then, any change to the kernel's
  // permutation semantics must be mirrored here.
  std::vector<int64_t> perm_local(static_cast<size_t>(T_local));
  if (!load_balance) {
    for (int64_t i = 0; i < T_local; ++i) {
      perm_local[static_cast<size_t>(i)] = rank * T_local + i;
    }
  } else {
    if (T_full % (2 * static_cast<int64_t>(world_size)) != 0) {
      throw std::invalid_argument(
          "shard_sequence_pre_embed: HeadTail requires T_full divisible by 2*world_size");
    }
    int64_t chunk_sz = T_full / (2 * static_cast<int64_t>(world_size));
    int64_t head_chunk = rank;
    int64_t tail_chunk = 2 * static_cast<int64_t>(world_size) - 1 - rank;
    for (int64_t i = 0; i < chunk_sz; ++i) {
      perm_local[static_cast<size_t>(i)] = head_chunk * chunk_sz + i;
      perm_local[static_cast<size_t>(chunk_sz + i)] = tail_chunk * chunk_sz + i;
    }
  }

  // pos_local: [1, T_local] int64 on device.
  Tensor pos_cpu(Shape{{1, T_local}}, TensorOptions().with_dtype(Dtype::Int64));
  std::memcpy(pos_cpu.data(), perm_local.data(),
              static_cast<size_t>(T_local) * sizeof(int64_t));
  Tensor pos_local = pos_cpu.to(device);

  ShardedInputs out;
  out.pos_local = pos_local;

  if (!load_balance) {
    // Contiguous: zero-copy view + contiguous materialization.
    auto idx_chunks = const_cast<Tensor &>(idx_full)
                          .make_shards_inplace_axis(
                              static_cast<size_t>(world_size), 1);
    out.idx_local = idx_chunks[static_cast<size_t>(rank)].contiguous();
    if (y_full.is_valid()) {
      auto y_chunks = const_cast<Tensor &>(y_full)
                          .make_shards_inplace_axis(
                              static_cast<size_t>(world_size), 1);
      out.y_local = y_chunks[static_cast<size_t>(rank)].contiguous();
    }
  } else {
    // HeadTail: build [B, T_local] gather index and gather along axis 1.
    Tensor gidx_cpu(Shape{{B, T_local}},
                    TensorOptions().with_dtype(Dtype::Int64));
    int64_t *gp = static_cast<int64_t *>(gidx_cpu.data());
    for (int64_t b = 0; b < B; ++b) {
      std::memcpy(gp + b * T_local, perm_local.data(),
                  static_cast<size_t>(T_local) * sizeof(int64_t));
    }
    Tensor gather_idx = gidx_cpu.to(device);
    out.idx_local = OwnTensor::gather(idx_full, /*dim=*/1, gather_idx);
    if (y_full.is_valid()) {
      out.y_local = OwnTensor::gather(y_full, /*dim=*/1, gather_idx);
    }
  }

  return out;
}

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
    //
    // Two independent flags now govern the LB pipeline:
    //   external_balanced  : pre-embedding HeadTail was already applied by
    //                        the caller (pre_sharded && load_balance_).
    //                        CP must NOT call its internal HeadTail kernel.
    //   sub_chunk_active   : round-robin Q/K/V sub-chunking + partial-merger
    //                        + pipelined dkv_rotater are active. Driven by
    //                        load_balance_ regardless of is_causal_, matching
    //                        PyTorch _attention.py @ e9ebbd3b.
    //
    // PyTorch dispatch table (enable_load_balance=True, is_causal=True):
    //   i == 0           : IS_CAUSAL,   full Q, full K/V, partial=false
    //   0 < i <= rank    : NOT_CAUSAL,  full Q, K/V[:T/2], partial=false
    //   i >  rank        : NOT_CAUSAL,  Q[T/2:], full K/V, partial=true
    //   never SKIP under LB.
    // HeadTail requires T divisible by 2*world_size (chunk-level layout has
    // chunk_sz = T/(2*N)). Generation paths pass variable, often-small T
    // (e.g. T=10 during incremental token generation) that may not satisfy
    // this. When the internal kernel would be invoked (pre_sharded=false)
    // and divisibility fails, fall back to a non-LB path for this call.
    // PyTorch has the same divisibility constraint.
    const int64_t T_full = q.shape().dims[2];
    const bool lb_div_ok = (T_full % (2 * static_cast<int64_t>(world_size_)) == 0);
    const bool lb_effective = load_balance_ && (pre_sharded || lb_div_ok);

    const bool external_balanced = pre_sharded && lb_effective;
    const bool sub_chunk_active = lb_effective;

    // Q can stay strided: sdpa_fused_forward / sdpa_fused_backward read
    // strides explicitly. Q is never sent across NCCL.
    //
    // K, V are kept contiguous: the ring rotator's cudaMemcpyAsync below
    // does a flat byte copy, and the strided pack-kernel alternative
    // (KVPackKernel) was measured slower than cudaMemcpyAsync on the
    // contig hot path. Keep contig here; pack kernel infrastructure remains
    // available for future use cases where source is genuinely strided.
    Tensor q_work = pre_sharded ? q : autograd::contiguous(q);
    Tensor k_work = autograd::contiguous(k);
    Tensor v_work = autograd::contiguous(v);

    if (lb_effective && !external_balanced) {
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
    std::vector<Tensor> saved_out_per_step(world_size_);

    // Sequence length of each rank's local chunk
    int64_t T_local_fwd = local_q.shape().dims[2];
    const int seq_dim = 2; // [B, H, T, D]

    // ---- Compute/comm overlap setup ----
    // CP_NO_OVERLAP=1 reverts to the serialized (CPU-blocking) path for A/B.
    // Value-aware: unset OR "0" => overlap ON; any other value => OFF.
    const char *_cp_no_ovl = std::getenv("CP_NO_OVERLAP");
    const bool OVERLAP =
        (_cp_no_ovl == nullptr) || (_cp_no_ovl[0] == '0' && _cp_no_ovl[1] == '\0');
    cudaStream_t compute_stream = OwnTensor::cuda::getCurrentStream();

    int64_t k_numel = local_k.numel();
    int64_t kv_numel = k_numel * 2;
    // [#6] Double-buffered send staging: each slot holds K AND V (kv_numel each).
    Tensor send_buf[2] = {Tensor::empty(Shape({{kv_numel}}), local_k.opts()),
                          Tensor::empty(Shape({{kv_numel}}), local_k.opts())};
    std::shared_ptr<Work> exch_work[2] = {nullptr, nullptr};  // [#1] per send slot
    cudaEvent_t pack_event = nullptr;
    if (OVERLAP) cudaEventCreateWithFlags(&pack_event, cudaEventDisableTiming);

    // Current K, V being processed (starts with local chunk)
    Tensor curr_k = local_k;
    Tensor curr_v = local_v;

    for (int i = 0; i < world_size_; ++i) {
      // Step 1: If not first iteration, get K,V from previous exchange.
      // GPU-side wait (no CPU stall) when overlapping; the recv was posted in
      // step i-1 and overlapped that step's SDPA.
      if (i > 0) {
        Tensor next_kv = OVERLAP
            ? kv_rotator->next_buffer_streamordered(compute_stream)
            : kv_rotator->next_buffer();
        Tensor kv_flat = next_kv.flatten();
        curr_k = kv_flat.narrow(0, 0, k_numel).reshape(local_k.shape());
        curr_v = kv_flat.narrow(0, k_numel, k_numel).reshape(local_v.shape());
      }

      // Step 2: Send current K,V to next rank (async; recv lands in the OTHER
      // ping-pong slot so it overlaps this step's compute without aliasing).
      if (i < (world_size_ - 1)) {
        int s = i & 1;
        // [#1] Before reusing send_buf[s] (last used at step i-2), ensure that
        // send has drained reading it. GPU-side; usually already complete.
        if (OVERLAP && exch_work[s]) exch_work[s]->streamWait(compute_stream);
        size_t k_bytes = static_cast<size_t>(k_numel) * sizeof(float);
        cudaMemcpyAsync(send_buf[s].data<float>(), curr_k.data<float>(),
                        k_bytes, cudaMemcpyDeviceToDevice, compute_stream);
        cudaMemcpyAsync(send_buf[s].data<float>() + k_numel,
                        curr_v.data<float>(), k_bytes, cudaMemcpyDeviceToDevice,
                        compute_stream);
        cudaEvent_t ev = nullptr;
        if (OVERLAP) { cudaEventRecord(pack_event, compute_stream); ev = pack_event; }
        nvtxRangePushA("CP.fwd.ring.exchange_buffers");
        exch_work[s] = kv_rotator->exchange_buffers(send_buf[s], ev);
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
        } else if (sub_chunk_active) {
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

      if (sub_chunk_active && i > 0) {
        if (i <= rank_) {
          // Past chunk with LB: full Q, first half of K/V.
          // sdpa_fused_forward reads per-tensor B/M/H strides from Q/K/V, so a
          // strided half-view from make_shards_inplace_axis can be passed
          // directly. Avoids the materialization copy that previously dominated
          // the LB strided_copy_kernel time in nsys.
          std::vector<Tensor> k_halves =
              curr_k.make_shards_inplace_axis(2, seq_dim);
          std::vector<Tensor> v_halves =
              curr_v.make_shards_inplace_axis(2, seq_dim);
          k_use = k_halves[0];
          v_use = v_halves[0];
        } else {
          // Future chunk with LB: second half of Q, full K/V.
          // Q strides are read from the tensor by the kernel — half-view OK.
          std::vector<Tensor> q_halves =
              local_q.make_shards_inplace_axis(2, seq_dim);
          q_use = q_halves[1];
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
      if (!sub_chunk_active && is_causal_) {
        int source_rank =
            ((rank_ - i) % world_size_ + world_size_) % world_size_;
        q_off = rank_ * static_cast<int>(T_local_fwd);
        k_off = source_rank * static_cast<int>(T_local_fwd);
      }
      // [KERNEL SHAPE CHECK] one-time, first few SDPA calls on rank 0: print the
      // ACTUAL q/k/v shapes fed to the attention kernel. Confirms the per-rank
      // local seqlen (and HeadTail sub-chunks) reaching the kernel, not full T.
      static int kshape_prints = 0;
      if (std::getenv("CP_DEBUG_SHAPES") && rank_ == 0 && kshape_prints < 4) {
        ++kshape_prints;
        std::cout << "[KERNEL SHAPE CHECK] ring_step i=" << i
                  << " q_use=[" << q_use.shape().dims[0] << ","
                  << q_use.shape().dims[1] << "," << q_use.shape().dims[2] << ","
                  << q_use.shape().dims[3] << "]"
                  << " k_use seqlen=" << k_use.shape().dims[2]
                  << " v_use seqlen=" << v_use.shape().dims[2]
                  << " causal=" << (use_causal ? 1 : 0)
                  << " partial=" << (use_partial ? 1 : 0) << "\n";
      }
      SDPAResult result = sdpa_fused_forward(
          q_use, k_use, v_use, use_causal, attn_scale_, q_off, k_off);

      // Save per-step LSE and OUT for backward. Backward calls
      // sdpa_fused_backward with these per-step values so the kernel's
      // D = sum(out*dout) and softmax(QK^T - lse) are computed against the
      // SAME K used in this step, not the final merged values (which
      // accumulate contributions from other ranks' K and corrupt the partial
      // step's gradient computation).
      saved_lse_per_step[i] = result.lse;
      saved_out_per_step[i] = result.out;

      // Step 6: Merge into accumulator (with partial flag)
      merger.step(result.out, result.lse, use_partial);

      // DUMP_CP_DEEP_FWD=1: per-step merged_out/merged_lse + per-step
      // SDPA result (block_out/block_lse). Lets us pin which ring step
      // (i=0 full merge or i>=1 partial-tail merge) introduces drift.
      {
        const char *env = std::getenv("DUMP_CP_DEEP_FWD");
        if (env && env[0] == '1') {
          auto [cur_out, cur_lse] = merger.results();
          auto save_bin = [&](const char *label, const Tensor &t) {
            Tensor host = t.to_cpu();
            std::string path = std::string("/tmp/cp_bwd_test/deep/cpp_") +
                               label + "_fwdstep" + std::to_string(i) +
                               "_rank" + std::to_string(rank_) + ".bin";
            std::ofstream fout(path, std::ios::binary);
            fout.write(reinterpret_cast<const char *>(host.data<float>()),
                       host.numel() * sizeof(float));
          };
          save_bin("block_out", result.out);
          save_bin("block_lse", result.lse);
          save_bin("merged_out", cur_out);
          save_bin("merged_lse", cur_lse);
        }
      }
    }

    if (pack_event) cudaEventDestroy(pack_event);

    // ----- Phase 3: Get final merged result -----
    auto [merged_out, merged_lse] = merger.results();

    // DUMP_CP_DEEP_FWD=1: save merged_out and merged_lse as .bin files for
    // PT-vs-C++ parity diff of the forward outputs that feed backward.
    {
      const char *env = std::getenv("DUMP_CP_DEEP_FWD");
      if (env && env[0] == '1') {
        auto save_bin = [&](const char *label, const Tensor &t) {
          Tensor host = t.to_cpu();
          std::string path = std::string("/tmp/cp_bwd_test/deep/cpp_") + label +
                             "_rank" + std::to_string(rank_) + ".bin";
          std::ofstream fout(path, std::ios::binary);
          fout.write(reinterpret_cast<const char *>(host.data<float>()),
                     host.numel() * sizeof(float));
        };
        save_bin("merged_out", merged_out);
        save_bin("merged_lse", merged_lse);
      }
    }

    // Optional first-call dump for PT parity probe (gated by DUMP_CP_OUT=1).
    // Only fires on the very first forward_cp invocation in this process,
    // which corresponds to block 0, step 0, micro 0.
    {
      static std::atomic<int> _cp_call_idx{0};
      int call_idx = _cp_call_idx.fetch_add(1);
      const char *env = std::getenv("DUMP_CP_OUT");
      if (call_idx == 0 && env && env[0] == '1') {
        Tensor host = merged_out.to_cpu();
        const float *p = host.data<float>();
        int run_idx = 0;
        while (true) {
          std::string probe = "block0_merged_out_rank" + std::to_string(rank_) +
                              "_" + std::to_string(run_idx) + ".md";
          std::ifstream check(probe);
          if (!check.good()) break;
          ++run_idx;
        }
        std::string path = "block0_merged_out_rank" + std::to_string(rank_) +
                           "_" + std::to_string(run_idx) + ".md";
        std::ofstream df(path);
        const auto &dims = merged_out.shape().dims;
        df << "shape=[" << dims[0] << "," << dims[1] << "," << dims[2] << ","
           << dims[3] << "]\n";
        df << "first16: [";
        int64_t n = std::min<int64_t>(16, merged_out.numel());
        for (int64_t i = 0; i < n; ++i) {
          df << p[i] << (i == n - 1 ? "" : ", ");
        }
        df << "]\n";
      }
    }

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

      // Undo internal HeadTail only when CP itself applied it. If the caller
      // pre-balanced the inputs externally (external_balanced), unshard output
      // is left in HeadTail order to match the caller's expected layout.
      if (lb_effective && !external_balanced) {
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
          saved_partial_flags, saved_lse_per_step, saved_out_per_step,
          merged_lse, merged_out.detach(), pg_, attn_scale_, is_causal_,
          rot_type, load_balance_, world_size_, rank_, unshard, recompute_k_,
          sub_chunk_active, external_balanced);

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
