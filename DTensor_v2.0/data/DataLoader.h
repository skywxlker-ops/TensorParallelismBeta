#ifndef DTENSOR_DATA_LOADER_H
#define DTENSOR_DATA_LOADER_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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

// Adjust path to point to TensorLib
#include "../Tensor-Implementations/include/TensorLib.h"

namespace fs = std::filesystem;

// Helper to list shards
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

// Memory-mapped shard view for uint16 tokens
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

        size_t total_tokens = file_bytes_ / 2;
        // std::cout << "Tokens: " << total_tokens << std::endl;
        tokens_ = std::min(total_tokens, max_tokens);

        // Map the whole file. OS will page-in only what you touch.
        data_ = ::mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            ::close(fd_); fd_ = -1; data_ = nullptr;
            throw std::runtime_error("mmap failed: " + path);
        }
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

    // Read tokens[start .. start+count) into out.
    void read_block(size_t start, size_t count, std::vector<u_int16_t>& out) const {
        if (start + count > tokens_) throw std::out_of_range("read_block out of range");
        out.resize(count);
        const u_int16_t* p = reinterpret_cast<const u_int16_t*>(data_);
        for (size_t i = 0; i < count; ++i) out[i] = p[start + i];
    }

private:
    std::string path_;
    int fd_ = -1;
    void* data_ = nullptr;
    size_t file_bytes_ = 0;
    size_t tokens_ = 0;
};

struct Batch {
    int B = 0, T = 0;
    // Flat row-major arrays of shape (B, T): index = b*T + t
    std::vector<u_int16_t> x;
    std::vector<u_int16_t> y;
    OwnTensor::Tensor input;
    OwnTensor::Tensor target;
};

class DataLoaderLite {
public:
    // Original Constructor (auto-scan)
    DataLoaderLite(int B, int T,
                     int rank, int world_size,
                     const std::string& split,
                     const std::string& data_root,
                     bool master_process = true,
                     size_t max_tokens_per_shard = 4000000000)
        : B_(B), T_(T),
          rank_(rank), world_(world_size),
          split_(split), root_(data_root),
          master_(master_process),
          max_tokens_(max_tokens_per_shard) {

        if (!(split_ == "train" || split_ == "val"))
            throw std::runtime_error("split must be 'train' or 'val'");
        validate_params();

        shards_ = list_shards(root_, split_, ".bin");
        if (shards_.empty())
            throw std::runtime_error("no .bin shards found for split " + split_ + " in " + root_);

        if (master_) {
            std::cout << "DataLoader: found " << shards_.size() << " shards for split " << split_ << "\n";
        }

        reset();
    }
    
    // NEW: Constructor with explicit file list
    DataLoaderLite(int B, int T,
                   int rank, int world_size,
                   const std::vector<std::string>& files,
                   bool master_process = true,
                   size_t max_tokens_per_shard = 4000000000)
        : B_(B), T_(T),
          rank_(rank), world_(world_size),
          split_("custom"), root_(""),
          master_(master_process),
          max_tokens_(max_tokens_per_shard),
          shards_(files) {
        
        validate_params();
        
        if (shards_.empty())
            throw std::runtime_error("DataLoader: provided file list is empty");
            
        // Check files exist
        for (const auto& f : shards_) {
            if (!fs::exists(f)) {
                 throw std::runtime_error("DataLoader: file not found: " + f);
            }
        }

        if (master_) {
            std::cout << "DataLoader: using " << shards_.size() << " explicit files\n";
            for(const auto& f : shards_) std::cout << "  - " << f << "\n";
        }

        reset();
    }

private:
    void validate_params() {
        if (B_ <= 0 || T_ <= 0)
            throw std::runtime_error("B and T must be > 0");
        if (world_ <= 0 || rank_ < 0 || rank_ >= world_)
            throw std::runtime_error("invalid rank/world_size");
    }

public:
    void reset() {
        current_shard_ = 0;
        shard_.open(shards_[current_shard_], max_tokens_);

        // current_position = B*T*process_rank
        pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);
    }


    Batch next_batch() {
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        const size_t need = BT + 1; // because y is shifted by 1

        // If the current shard doesn't even have enough tokens for one batch from start, that's an edge case,
        // but generally we check if pos_ + need exceeds the shard.

        if (pos_ + need > shard_.size_tokens()) {
            advance_shard();
        }

        std::vector<u_int16_t> buf;
        shard_.read_block(pos_, need, buf);

        Batch b;
        b.B = B_; b.T = T_;
        b.x.resize(BT);
        b.y.resize(BT);

        for (size_t i = 0; i < BT; ++i) {
            b.x[i] = buf[i];
            b.y[i] = buf[i + 1];
        }
        
        // Convert to OwnTensor on CPU first if needed, or directly device
        // The original code created CUDA tensors immediately.
        // We will keep that behavior but ensure we have a current device set, or explicit device.
        
        b.input = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, {OwnTensor::Dtype::UInt16, OwnTensor::Device::CUDA});
        b.input.set_data(b.x);

        b.target = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, {OwnTensor::Dtype::UInt16, OwnTensor::Device::CUDA});
        b.target.set_data(b.y);

        // Update position for next batch
        // Each process advances by B*T*world_size
        pos_ += BT * static_cast<size_t>(world_);

        // Check if next batch would go OOB
        if (pos_ + (BT * static_cast<size_t>(world_) + 1) > shard_.size_tokens()) {
            advance_shard();
        }

        return b;
    }

private:
    void advance_shard() {
        current_shard_ = (current_shard_ + 1) % shards_.size();
        shard_.open(shards_[current_shard_], max_tokens_);

        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        // Reset position for the new shard
        pos_ = BT * static_cast<size_t>(rank_);
        
        if (master_) {
            // std::cout << "DataLoader: advanced to shard " << shards_[current_shard_] << "\n";
        }
    }

    int B_, T_;
    int rank_, world_;
    std::string split_, root_;
    bool master_;
    size_t max_tokens_;

    std::vector<std::string> shards_;
    size_t current_shard_ = 0;
    size_t pos_ = 0;

    UInt16ShardView shard_;
};

#endif // DTENSOR_DATA_LOADER_H
