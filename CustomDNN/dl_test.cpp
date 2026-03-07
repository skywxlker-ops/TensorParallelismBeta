#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <string>
#include <vector>
#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "TensorLib.h"

namespace fs = std::filesystem;



static int getenv_int(const char* key, int def) {
    const char* v = std::getenv(key);
    return v ? std::atoi(v) : def;
}

static std::vector<std::string> list_shards(const std::string& root,
                                            const std::string& split,
                                            const std::string& ext = ".bin") {
    std::vector<std::string> shards;
    for (const auto& e : fs::directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        std::string name = p.filename().string();
        if (p.extension() == ext && name.find(split) != std::string::npos) {
            shards.push_back(p.string());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

// mmap shard view 

class UInt16ShardView {
public:
    UInt16ShardView() = default;
    ~UInt16ShardView() { close(); }

    UInt16ShardView(const UInt16ShardView&) = delete;
    UInt16ShardView& operator=(const UInt16ShardView&) = delete;

    void open(const std::string& path, size_t max_tokens) {
        close();
        path_ = path;

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("failed to open: " + path);

        struct stat st {};
        if (fstat(fd_, &st) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("failed to stat: " + path);
        }

        file_bytes_ = static_cast<size_t>(st.st_size);
        if (file_bytes_ % sizeof(u_int16_t) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("file size not divisible by 2 (uint16): " + path);
        }

        size_t total_tokens = file_bytes_ /2;
        tokens_ = std::min(total_tokens, max_tokens);

        // Map the whole file. OS will page-in only what you touch.
        data_ = ::mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            ::close(fd_); fd_ = -1; data_ = nullptr;
            throw std::runtime_error("mmap failed: " + path);
        }

        // Optimization #4: hint the kernel we'll read sequentially
        ::madvise(data_, file_bytes_, MADV_SEQUENTIAL);
    }

    void close() {
        if (data_) {
            ::munmap(data_, file_bytes_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        file_bytes_ = 0;
        tokens_ = 0;
        path_.clear();
    }

    size_t size_tokens() const { return tokens_; }
    const std::string& path() const { return path_; }

    // Optimization #2: direct pointer into the mmap'd region (zero-copy access)
    const u_int16_t* data_ptr() const {
        return reinterpret_cast<const u_int16_t*>(data_);
    }

    // Optimization #1: memcpy instead of element-by-element loop
    void read_block(size_t start, size_t count, std::vector<u_int16_t>& out) const {
        if (start + count > tokens_) throw std::out_of_range("read_block out of range");
        out.resize(count);
        const u_int16_t* p = reinterpret_cast<const u_int16_t*>(data_);
        std::memcpy(out.data(), p + start, count * sizeof(u_int16_t));
    }

private:
    std::string path_;
    int fd_ = -1;
    void* data_ = nullptr;
    size_t file_bytes_ = 0;
    size_t tokens_ = 0;
};


// Optimization #6: removed dead x/y vectors from Batch
struct Batch {
    int B = 0, T = 0;
    OwnTensor::Tensor input;
    OwnTensor::Tensor target;
};

class DataLoaderLite {
public:
    DataLoaderLite(int B, int T,
                     int rank, int world_size,
                     const std::string& split,
                     const std::string& data_root,
                     bool master_process = true,
                     size_t max_tokens_per_shard = 100000000,
                     int device_idx = 0)
        : B_(B), T_(T),
          rank_(rank), world_(world_size),
          split_(split), root_(data_root),
          master_(master_process),
          max_tokens_(max_tokens_per_shard),
          device_idx_(device_idx) {

        if (!(split_ == "train" || split_ == "val"))
            throw std::runtime_error("split must be 'train' or 'val'");
        if (B_ <= 0 || T_ <= 0)
            throw std::runtime_error("B and T must be > 0");
        if (world_ <= 0 || rank_ < 0 || rank_ >= world_)
            throw std::runtime_error("invalid rank/world_size");

        shards_ = list_shards(root_, split_, ".bin");
        if (shards_.empty())
            throw std::runtime_error("no .bin shards found for split " + split_);

        if (master_) {
            std::cout << "found " << shards_.size() << " shards for split " << split_ << "\n";
        }

        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        OwnTensor::DeviceIndex dev(OwnTensor::Device::CUDA, device_idx_);

        // Double-buffered pinned staging + GPU tensors for async H2D overlap.
        // Buffer 0 and buffer 1 alternate: one is consumed by compute while
        // the other receives the next batch's H2D copy on copy_stream_.
        for (int i = 0; i < 2; ++i) {
            pinned_x_[i] = nullptr;
            pinned_y_[i] = nullptr;
            cudaError_t e1 = cudaHostAlloc(&pinned_x_[i], BT * sizeof(u_int16_t), cudaHostAllocDefault);
            cudaError_t e2 = cudaHostAlloc(&pinned_y_[i], BT * sizeof(u_int16_t), cudaHostAllocDefault);
            if (e1 != cudaSuccess || e2 != cudaSuccess) {
                std::cerr << "Warning: cudaHostAlloc failed for buffer " << i << ", falling back to pageable\n";
                if (pinned_x_[i]) { cudaFreeHost(pinned_x_[i]); pinned_x_[i] = nullptr; }
                if (pinned_y_[i]) { cudaFreeHost(pinned_y_[i]); pinned_y_[i] = nullptr; }
            }

            gpu_input_[i] = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}},
                OwnTensor::TensorOptions().with_dtype(OwnTensor::Dtype::UInt16).with_device(dev));
            gpu_target_[i] = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}},
                OwnTensor::TensorOptions().with_dtype(OwnTensor::Dtype::UInt16).with_device(dev));
        }

        cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&copy_done_, cudaEventDisableTiming);

        reset();
    }

    ~DataLoaderLite() {
        for (int i = 0; i < 2; ++i) {
            if (pinned_x_[i]) { cudaFreeHost(pinned_x_[i]); }
            if (pinned_y_[i]) { cudaFreeHost(pinned_y_[i]); }
        }
        cudaStreamDestroy(copy_stream_);
        cudaEventDestroy(copy_done_);
    }

    // Non-copyable (pinned buffer ownership)
    DataLoaderLite(const DataLoaderLite&) = delete;
    DataLoaderLite& operator=(const DataLoaderLite&) = delete;

    void reset() {
        current_shard_ = 0;
        shard_.open(shards_[current_shard_], max_tokens_);
        pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);
        buf_idx_ = 0;
        prefetched_ = false;
    }

    // Original synchronous next_batch (for validation or non-overlapped paths)
    Batch next_batch() {
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        const size_t need = BT + 1;

        if (shard_.size_tokens() < need) {
            throw std::runtime_error("shard too small for one batch: " + shard_.path());
        }

        if (pos_ + need > shard_.size_tokens()) {
            advance_shard();
        }

        const u_int16_t* tokens = shard_.data_ptr() + pos_;
        int buf = buf_idx_;

        Batch b;
        b.B = B_; b.T = T_;

        if (pinned_x_[buf]) {
            std::memcpy(pinned_x_[buf], tokens, BT * sizeof(u_int16_t));
            std::memcpy(pinned_y_[buf], tokens + 1, BT * sizeof(u_int16_t));
            cudaMemcpy(gpu_input_[buf].data(), pinned_x_[buf], BT * sizeof(u_int16_t), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_target_[buf].data(), pinned_y_[buf], BT * sizeof(u_int16_t), cudaMemcpyHostToDevice);
        } else {
            x_buf_.resize(BT);
            y_buf_.resize(BT);
            std::memcpy(x_buf_.data(), tokens, BT * sizeof(u_int16_t));
            std::memcpy(y_buf_.data(), tokens + 1, BT * sizeof(u_int16_t));
            gpu_input_[buf].set_data(x_buf_);
            gpu_target_[buf].set_data(y_buf_);
        }

        b.input = gpu_input_[buf];
        b.target = gpu_target_[buf];

        pos_ += BT * static_cast<size_t>(world_);
        if (pos_ + (BT * static_cast<size_t>(world_) + 1) > shard_.size_tokens()) {
            advance_shard();
        }

        buf_idx_ ^= 1;
        return b;
    }

    // =========================================================================
    // Async double-buffered pipeline
    // =========================================================================

    // Launch H2D copy for the next batch on copy_stream_.
    // Returns immediately — the copy runs concurrently with compute on stream 0.
    void prefetch_next_batch() {
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        const size_t need = BT + 1;

        if (pos_ + need > shard_.size_tokens()) {
            advance_shard();
        }

        const u_int16_t* tokens = shard_.data_ptr() + pos_;
        int buf = buf_idx_;   // the "back" buffer (not being consumed by compute)

        if (pinned_x_[buf]) {
            // Stage mmap → pinned (CPU memcpy, very fast for pinned memory)
            std::memcpy(pinned_x_[buf], tokens, BT * sizeof(u_int16_t));
            std::memcpy(pinned_y_[buf], tokens + 1, BT * sizeof(u_int16_t));

            // Async H2D on dedicated copy stream — overlaps with compute on stream 0
            cudaMemcpyAsync(gpu_input_[buf].data(), pinned_x_[buf],
                            BT * sizeof(u_int16_t), cudaMemcpyHostToDevice, copy_stream_);
            cudaMemcpyAsync(gpu_target_[buf].data(), pinned_y_[buf],
                            BT * sizeof(u_int16_t), cudaMemcpyHostToDevice, copy_stream_);
        } else {
            // Fallback: synchronous copy (no pinned memory available)
            x_buf_.resize(BT);
            y_buf_.resize(BT);
            std::memcpy(x_buf_.data(), tokens, BT * sizeof(u_int16_t));
            std::memcpy(y_buf_.data(), tokens + 1, BT * sizeof(u_int16_t));
            gpu_input_[buf].set_data(x_buf_);
            gpu_target_[buf].set_data(y_buf_);
        }

        // Record event so consumers can wait for this copy to finish
        cudaEventRecord(copy_done_, copy_stream_);

        pos_ += BT * static_cast<size_t>(world_);
        if (pos_ + (BT * static_cast<size_t>(world_) + 1) > shard_.size_tokens()) {
            advance_shard();
        }

        prefetched_ = true;
    }

    // Wait for the prefetched copy and return the batch.
    // After this call, the returned tensors are safe to read on any stream.
    Batch consume_prefetched() {
        if (!prefetched_) {
            throw std::runtime_error("consume_prefetched called without prefetch_next_batch");
        }

        // GPU-side wait: make stream 0 wait for the copy to finish.
        // CPU is NOT blocked — just inserts a dependency on the GPU.
        cudaStreamWaitEvent(0, copy_done_, 0);

        int buf = buf_idx_;
        Batch b;
        b.B = B_; b.T = T_;
        b.input = gpu_input_[buf];
        b.target = gpu_target_[buf];

        buf_idx_ ^= 1;    // swap to other buffer for next prefetch
        prefetched_ = false;
        return b;
    }

private:
    void advance_shard() {
        current_shard_ = (current_shard_ + 1) % shards_.size();
        std::cout << "Current shard: "<< shards_[current_shard_] << std::endl;
        std::cout << max_tokens_ << std::endl;
        shard_.open(shards_[current_shard_], max_tokens_);

        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        pos_ = BT * static_cast<size_t>(rank_);
    }

    int B_, T_;
    int rank_, world_;
    std::string split_, root_;
    bool master_;
    size_t max_tokens_;
    int device_idx_;

    std::vector<std::string> shards_;
    size_t current_shard_ = 0;
    size_t pos_ = 0;

    UInt16ShardView shard_;

    // Double-buffered pinned staging + GPU tensors (indices 0 and 1)
    u_int16_t* pinned_x_[2] = {};
    u_int16_t* pinned_y_[2] = {};
    OwnTensor::Tensor gpu_input_[2];
    OwnTensor::Tensor gpu_target_[2];
    int buf_idx_ = 0;         // which buffer to write next

    // Dedicated copy stream + completion event for async H2D
    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t copy_done_ = nullptr;
    bool prefetched_ = false;

    // Fallback pageable buffers (only used if pinned alloc fails)
    std::vector<u_int16_t> x_buf_;
    std::vector<u_int16_t> y_buf_;
};

