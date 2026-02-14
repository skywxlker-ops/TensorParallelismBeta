#include "allocator.h"
#include "block.h"
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <stdexcept>

class SimpleCachingAllocator : public MemoryAllocator {
public:
    SimpleCachingAllocator() : total_allocated_(0), total_freed_(0) {}

    void* allocate(size_t size, cudaStream_t stream = 0) override {
        std::lock_guard<std::mutex> lock(mutex_);
        void* dev_ptr = nullptr;
        cudaError_t err = cudaMalloc(&dev_ptr, size);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));

        MemoryBlock block(dev_ptr, size, stream);
        active_blocks_[dev_ptr] = block;
        total_allocated_ += size;

        std::cout << "[Allocator] Allocated " << size << " bytes at " << dev_ptr << std::endl;
        return dev_ptr;
    }

    void free(void* ptr, cudaStream_t stream = 0) override {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = active_blocks_.find(ptr);
        if (it == active_blocks_.end()) {
            std::cerr << "[Allocator] Unknown pointer " << ptr << std::endl;
            cudaFree(ptr);
            return;
        }
        cudaFree(ptr);
        total_freed_ += it->second.size;
        std::cout << "[Allocator] Freed " << it->second.size << " bytes at " << ptr << std::endl;
        active_blocks_.erase(it);
    }

    void emptyCache() override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [ptr, block] : active_blocks_)
            cudaFree(ptr);
        active_blocks_.clear();
        std::cout << "[Allocator] Cache emptied." << std::endl;
    }

    void printStats() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "===== Allocator Stats =====\n";
        std::cout << "Active blocks: " << active_blocks_.size() << "\n";
        std::cout << "Total allocated: " << total_allocated_ << " bytes\n";
        std::cout << "Total freed: " << total_freed_ << " bytes\n";
        std::cout << "===========================\n";
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<void*, MemoryBlock> active_blocks_;
    size_t total_allocated_, total_freed_;
};

std::unique_ptr<MemoryAllocator> global_allocator = std::make_unique<SimpleCachingAllocator>();
