#pragma once

#include "process_group/ProcessGroupNCCL.h"
#include "core/Tensor.h"
#include <vector>
#include <memory>
#include <stdexcept>

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

    virtual void exchange_buffers(Tensor& curr_buffer) = 0;
    virtual Tensor next_buffer() = 0;

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
        : RingRotatorBase(pg), recv_buffer_(), buffer_allocated_(false) {}

    void exchange_buffers(Tensor& curr_buffer) override {
        int next_rank = (rank_ + 1) % world_size_;
        int prev_rank = (rank_ - 1 + world_size_) % world_size_;

        size_t count = static_cast<size_t>(curr_buffer.numel());
        Dtype dtype = curr_buffer.dtype();

        // Pre-allocate receive buffer once, reuse across ring steps
        if (!buffer_allocated_) {
            recv_buffer_ = Tensor::empty(curr_buffer.shape(), curr_buffer.opts());
            buffer_allocated_ = true;
        }

        pending_work_ = pg_->sendrecv_async(
            curr_buffer.data<float>(), recv_buffer_.data<float>(),
            next_rank, prev_rank, count, dtype);
    }

    Tensor next_buffer() override {
        if (pending_work_) {
            pending_work_->wait();
            pending_work_ = nullptr;
        }

        if (!recv_buffer_.is_valid()) {
            throw std::runtime_error("P2PRingRotator::next_buffer: no buffer available");
        }
        return recv_buffer_;
    }

private:
    Tensor recv_buffer_;
    bool buffer_allocated_;
    std::shared_ptr<Work> pending_work_;
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
    void exchange_buffers(Tensor& curr_buffer) override {
        int next_rank = (rank_ + 1) % world_size_;
        int prev_rank = (rank_ - 1 + world_size_) % world_size_;

        size_t numel = static_cast<size_t>(curr_buffer.numel());
        Dtype dtype = curr_buffer.dtype();

        if (!buffer_allocated_) {
            recv_buffer_ = Tensor::empty(curr_buffer.shape(), curr_buffer.opts());
            // Build sparse split sizes: send numel to next_rank, recv numel from prev_rank
            sendcounts_.assign(world_size_, 0);
            recvcounts_.assign(world_size_, 0);
            senddispls_.assign(world_size_, 0);
            recvdispls_.assign(world_size_, 0);
            sendcounts_[next_rank] = numel;
            recvcounts_[prev_rank] = numel;
            // Displacements are 0 for both (single contiguous buffer)
            buffer_allocated_ = true;
        }

        pending_work_ = pg_->alltoallv_async(
            curr_buffer.data(), sendcounts_.data(), senddispls_.data(),
            recv_buffer_.data(), recvcounts_.data(), recvdispls_.data(),
            dtype);
    }

    Tensor next_buffer() override {
        if (pending_work_) {
            pending_work_->wait();
            pending_work_ = nullptr;
        }
        return recv_buffer_;
    }

private:
    Tensor recv_buffer_;
    bool buffer_allocated_;
    std::vector<size_t> sendcounts_;
    std::vector<size_t> recvcounts_;
    std::vector<size_t> senddispls_;
    std::vector<size_t> recvdispls_;
    std::shared_ptr<Work> pending_work_;
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

    void exchange_buffers(Tensor& curr_buffer) override {
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