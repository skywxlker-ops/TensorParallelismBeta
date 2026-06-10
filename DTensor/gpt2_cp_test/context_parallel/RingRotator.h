#pragma once

#include "process_group/ProcessGroupNCCL.h"
#include "core/Tensor.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// Base class for all ring rotators.
// A rotator shifts KV (or grad) buffers around a ring of ranks.
//
// Usage pattern (per ring iteration):
//   1. exchange_buffers(curr_buffer)   -- initiates async send of curr_buffer
//   2. next_buffer()                   -- blocks until the received buffer is ready
// ---------------------------------------------------------------------------
class RingRotatorBase {
public:
    RingRotatorBase(std::shared_ptr<ProcessGroupNCCL> pg)
        : pg_(pg),
          rank_(pg->get_rank()),
          world_size_(pg->get_worldsize()) {}

    virtual ~RingRotatorBase() = default;

    // Posts the async send of curr_buffer and receives into the rotator's
    // (double-buffered) recv slot. Returns the Work handle so the CALLER can
    // guard reuse of its own send-staging buffer (must outlive the transfer).
    //
    // pack_event: if non-null, the rotator makes its communication stream wait
    // on this event BEFORE issuing the NCCL send — used to order a caller-side
    // pack memcpy (on the compute stream) -> send (on the comm stream), so the
    // send never reads a half-packed buffer. nullptr => legacy behavior.
    virtual std::shared_ptr<Work> exchange_buffers(Tensor& curr_buffer,
                                                   cudaEvent_t pack_event = nullptr) = 0;

    // CPU-blocking consume (legacy path; used when CP_NO_OVERLAP is set).
    virtual Tensor next_buffer() = 0;

    // GPU-side consume: make `compute_stream` wait on the pending recv event
    // (no CPU stall) and return the received buffer. Default falls back to the
    // CPU-blocking next_buffer() (correct, just not overlapped) so rotators that
    // cannot pipeline (e.g. AllGather) remain valid.
    virtual Tensor next_buffer_streamordered(cudaStream_t /*compute_stream*/) {
        return next_buffer();
    }

protected:
    std::shared_ptr<ProcessGroupNCCL> pg_;
    int rank_;
    int world_size_;
};


// ---------------------------------------------------------------------------
// P2P Ring Rotator (optimized)
//
// Pre-allocates the receive buffer on first use to avoid per-step allocation.
// Uses ncclSend / ncclRecv point-to-point operations.
// Even ranks send first then receive; odd ranks receive first then send.
// ---------------------------------------------------------------------------
class P2PRingRotator : public RingRotatorBase {
public:
    P2PRingRotator(std::shared_ptr<ProcessGroupNCCL> pg)
        : RingRotatorBase(pg), buffer_allocated_(false), slot_(0) {}

    // Double-buffered ring shift: receive into the slot NOT being read by the
    // current step's compute, so recv(i+1) can be in flight while compute(i)
    // reads recv(i). Returns the Work covering both this step's send and recv.
    std::shared_ptr<Work> exchange_buffers(Tensor& curr_buffer,
                                           cudaEvent_t pack_event = nullptr) override {
        int next_rank = (rank_ + 1) % world_size_;
        int prev_rank = (rank_ - 1 + world_size_) % world_size_;

        size_t count = static_cast<size_t>(curr_buffer.numel());
        Dtype dtype = curr_buffer.dtype();

        // Pre-allocate BOTH receive slots once (ping-pong), reuse across steps.
        if (!buffer_allocated_) {
            for (int s = 0; s < 2; ++s)
                recv_[s] = Tensor::empty(curr_buffer.shape(), curr_buffer.opts());
            buffer_allocated_ = true;
        }

        // Order send-after-pack: make the comm stream wait for the caller's pack
        // memcpy (recorded on the compute stream) before NCCL reads curr_buffer.
        if (pack_event != nullptr) {
            cudaStreamWaitEvent(pg_->getStream(), pack_event, 0);
        }

        slot_ ^= 1;  // receive into the free slot
        work_[slot_] = pg_->sendrecv_async(
            curr_buffer.data<float>(), recv_[slot_].data<float>(),
            next_rank, prev_rank, count, dtype);
        return work_[slot_];
    }

    Tensor next_buffer() override {
        if (work_[slot_]) {
            work_[slot_]->wait();      // CPU-blocking (legacy / CP_NO_OVERLAP path)
            work_[slot_] = nullptr;
        }
        if (!recv_[slot_].is_valid()) {
            throw std::runtime_error("P2PRingRotator::next_buffer: no buffer available");
        }
        return recv_[slot_];
    }

    Tensor next_buffer_streamordered(cudaStream_t compute_stream) override {
        if (work_[slot_]) {
            work_[slot_]->streamWait(compute_stream);  // GPU-side, no CPU stall
        }
        if (!recv_[slot_].is_valid()) {
            throw std::runtime_error(
                "P2PRingRotator::next_buffer_streamordered: no buffer available");
        }
        return recv_[slot_];
    }

private:
    Tensor recv_[2];                       // ping-pong recv slots
    bool buffer_allocated_;
    int  slot_;                            // slot most recently received into
    std::shared_ptr<Work> work_[2];        // pending op per recv slot
};


// ---------------------------------------------------------------------------
// AlltoAll Ring Rotator
//
// Implements ring rotation via ncclAlltoAll using the shifted permutation
// pattern dsts = [1, 2, ..., n-1, 0].
//
// Each rank builds a send buffer with world_size chunks. Only the chunk
// destined for next_rank contains the actual KV data; all other chunks
// are zeroed. After the collective, we extract the data from the
// prev_rank slot of the receive buffer.
//
// This is deadlock-free and topology-aware (NCCL optimizes the routing).
// ---------------------------------------------------------------------------
class AlltoAllRingRotator : public RingRotatorBase {
public:
    AlltoAllRingRotator(std::shared_ptr<ProcessGroupNCCL> pg)
        : RingRotatorBase(pg), buffer_allocated_(false) {}

    // Sparse alltoall ring shift: send curr_buffer to rank (i+1)%N,
    // receive from rank (i-1)%N. Matches PyTorch's permute_tensor
    // with dsts=[1,2,...,n-1,0] using alltoallv with sparse split sizes.
    // Buffer: 1x per_rank_count (not world_size * per_rank_count).
    // One sparse alltoallv per step (send->next_rank, recv<-prev_rank): exactly
    // ONE in-flight op writing ONE recv destination per step — structurally
    // identical to P2P, so the ping-pong (recv_[2]/slot_) applies. The sparse
    // split arrays are slot-independent (they encode ranks, computed once).
    std::shared_ptr<Work> exchange_buffers(Tensor& curr_buffer,
                                           cudaEvent_t pack_event = nullptr) override {
        int next_rank = (rank_ + 1) % world_size_;
        int prev_rank = (rank_ - 1 + world_size_) % world_size_;

        size_t numel = static_cast<size_t>(curr_buffer.numel());
        Dtype dtype = curr_buffer.dtype();

        if (!buffer_allocated_) {
            for (int s = 0; s < 2; ++s)
                recv_[s] = Tensor::empty(curr_buffer.shape(), curr_buffer.opts());
            sendcounts_.assign(world_size_, 0);
            recvcounts_.assign(world_size_, 0);
            senddispls_.assign(world_size_, 0);
            recvdispls_.assign(world_size_, 0);
            sendcounts_[next_rank] = numel;
            recvcounts_[prev_rank] = numel;
            buffer_allocated_ = true;
        }

        if (pack_event != nullptr) {
            cudaStreamWaitEvent(pg_->getStream(), pack_event, 0);
        }

        slot_ ^= 1;
        work_[slot_] = pg_->alltoallv_async(
            curr_buffer.data(), sendcounts_.data(), senddispls_.data(),
            recv_[slot_].data(), recvcounts_.data(), recvdispls_.data(),
            dtype);
        return work_[slot_];
    }

    Tensor next_buffer() override {
        if (work_[slot_]) { work_[slot_]->wait(); work_[slot_] = nullptr; }
        return recv_[slot_];
    }

    Tensor next_buffer_streamordered(cudaStream_t compute_stream) override {
        if (work_[slot_]) work_[slot_]->streamWait(compute_stream);
        return recv_[slot_];
    }

private:
    Tensor recv_[2];
    bool buffer_allocated_;
    int  slot_ = 0;
    std::vector<size_t> sendcounts_;
    std::vector<size_t> recvcounts_;
    std::vector<size_t> senddispls_;
    std::vector<size_t> recvdispls_;
    std::shared_ptr<Work> work_[2];
};


// ---------------------------------------------------------------------------
// AllGather Ring Rotator
//
// Gathers all buffers from all ranks in a single all_gather call on the
// first exchange. Subsequent calls just index into the gathered buffer.
// ---------------------------------------------------------------------------
class AllGatherRingRotator : public RingRotatorBase {
public:
    AllGatherRingRotator(std::shared_ptr<ProcessGroupNCCL> pg)
        : RingRotatorBase(pg), idx_(0), aggregated_buffer_() {}

    std::shared_ptr<Work> exchange_buffers(Tensor& curr_buffer,
                                           cudaEvent_t /*pack_event*/ = nullptr) override {
        // [#8] AllGather cannot pipeline (single blocking collective); surface
        // the lost overlap once, unless the caller explicitly disabled overlap.
        static bool warned = false;
        if (!warned && std::getenv("CP_NO_OVERLAP") == nullptr) {
            warned = true;
            fprintf(stderr,
                    "[CP overlap WARNING] AllGatherRingRotator selected: ring "
                    "communication will NOT overlap compute (single blocking "
                    "all_gather). Use P2P/AlltoAll rotator for overlap.\n");
        }
        idx_ += 1;

        if (!aggregated_buffer_.is_valid()) {
            size_t per_rank_count = static_cast<size_t>(curr_buffer.numel());
            size_t total_count = per_rank_count * static_cast<size_t>(world_size_);
            Dtype dtype = curr_buffer.dtype();

            Shape agg_shape({{static_cast<int64_t>(total_count)}});
            aggregated_buffer_ = Tensor::empty(agg_shape, curr_buffer.opts());

            Tensor flat_input = curr_buffer.flatten();

            pg_->all_gather(
                flat_input.data<float>(),
                aggregated_buffer_.data<float>(),
                per_rank_count,
                dtype,
                true);

            per_rank_numel_ = per_rank_count;
        }
        return nullptr;  // blocking all_gather already completed; no pending Work
    }

    Tensor next_buffer() override {
        if (!aggregated_buffer_.is_valid()) {
            throw std::runtime_error("AllGatherRingRotator::next_buffer: exchange_buffers not called");
        }

        int source_rank = ((rank_ - idx_) % world_size_ + world_size_) % world_size_;
        int64_t offset = static_cast<int64_t>(source_rank) * static_cast<int64_t>(per_rank_numel_);

        float* base_ptr = aggregated_buffer_.data<float>() + offset;

        Shape chunk_shape({{static_cast<int64_t>(per_rank_numel_)}});
        Tensor chunk = Tensor::empty(chunk_shape, aggregated_buffer_.opts());

        cudaMemcpyAsync(
            chunk.data<float>(),
            base_ptr,
            per_rank_numel_ * sizeof(float),
            cudaMemcpyDeviceToDevice,
            0);

        return chunk;
    }

private:
    int idx_;
    Tensor aggregated_buffer_;
    size_t per_rank_numel_ = 0;
};