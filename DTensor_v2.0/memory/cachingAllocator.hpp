#pragma once
#include <cstddef>
#include <unordered_map>
#include <set>
#include <mutex>
#include <atomic>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

enum class Device { CPU, CUDA };
enum class PoolType { SMALL, LARGE };

struct Block {
    void* addr;
    size_t size;
    bool active;
    PoolType pool_type;
    Block* prev;
    Block* next;
    cudaStream_t stream;

    Block(void* a = nullptr, size_t s = 0, PoolType p = PoolType::SMALL, cudaStream_t str = 0)
        : addr(a), size(s), active(false), pool_type(p),
          prev(nullptr), next(nullptr), stream(str) {}
};

static constexpr size_t ONE_MB = 1024UL * 1024UL;
static constexpr size_t ALIGNMENT = 512;
static constexpr size_t MIN_SPLIT = 512;

// CUDA stream hashing helpers
struct StreamHasher {
    std::size_t operator()(const cudaStream_t &s) const noexcept {
        return std::hash<std::uintptr_t>()(reinterpret_cast<std::uintptr_t>(s));
    }
};
struct StreamEq {
    bool operator()(const cudaStream_t &a, const cudaStream_t &b) const noexcept {
        return reinterpret_cast<std::uintptr_t>(a) == reinterpret_cast<std::uintptr_t>(b);
    }
};

// Block size comparator (for best-fit multiset)
struct CompareBySize {
    bool operator()(const Block* a, const Block* b) const {
        if (a->size != b->size) return a->size < b->size;
        return std::uintptr_t(a->addr) < std::uintptr_t(b->addr);
    }
};

struct StreamInternal {
    std::multiset<Block*, CompareBySize> small_pool;
    std::multiset<Block*, CompareBySize> large_pool;
    std::vector<Block*> all_blocks;
};

class CachingAllocator {
public:
    CachingAllocator();
    ~CachingAllocator();

    Block* allocateMemory(size_t size, cudaStream_t stream);
    void freeMemory(Block* block);
    Block* requestMemory(size_t size, PoolType pool_type, cudaStream_t stream);
    void emptyCache(Device device = Device::CPU);

    size_t memoryAllocated() const;
    size_t memoryFree() const;
    void printStats() const;
    bool isAllocated(Block* block) const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<cudaStream_t, StreamInternal, StreamHasher, StreamEq> stream_to_cache_;
    std::atomic<size_t> total_allocated_;
    std::atomic<size_t> total_free_;

    size_t roundMemorySize(size_t size) const;
    PoolType selectPoolType(size_t size) const;
    Block* allocNewBlock(size_t size, cudaStream_t stream, PoolType pool_type);
    Block* findBestFit(std::multiset<Block*, CompareBySize>& pool, size_t size);
    void mergeAdjacent(Block* block, StreamInternal& internal);
};
