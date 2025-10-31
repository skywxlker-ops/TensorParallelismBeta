// #pragma once
// #include <cstddef>
// #include <unordered_map>
// #include <set>
// #include <mutex>
// #include <atomic>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>

// /*
//  * Production-Level FSDP Caching Allocator
//  * 
//  * Features:
//  * - Stream-aware, thread-safe
//  * - Best-fit allocation with splitting/merging
//  * - Lazy GPU allocation
//  * - Memory rounding & alignment
//  * - Detailed memory statistics
//  */

// enum class Device { CPU, CUDA };
// enum class PoolType { SMALL, LARGE };




// struct Block {
//     void* addr;            // Device pointer
//     size_t size;           // Size in bytes
//     bool active;           // In-use or free
//     PoolType pool_type;    // Pool type
//     Block* prev;           // For merge/split
//     Block* next;
//     cudaStream_t stream;   // Associated CUDA stream

//     Block(void* a = nullptr, size_t s = 0, PoolType p = PoolType::SMALL, cudaStream_t str = 0)
//         : addr(a), size(s), active(false), pool_type(p), prev(nullptr), next(nullptr), stream(str) {}
// };

// static constexpr size_t ONE_MB = 1024UL * 1024UL;
// static constexpr size_t ALIGNMENT = 512; // bytes
// static constexpr size_t MIN_SPLIT = 512; // minimum bytes to remain to justify splitting

// // Hash / equality for cudaStream_t to use in unordered_map
// struct StreamHasher {
//     //this makes the cudastream_t as a key to look into the map
//     std::size_t operator()(const cudaStream_t &s) const noexcept {
//         return std::hash<std::uintptr_t>()(reinterpret_cast<std::uintptr_t>(s));
//     }
// };


// //to check if two streams are equal.
// struct StreamEq {
//     bool operator()(const cudaStream_t &a, const cudaStream_t &b) const noexcept {
//         return reinterpret_cast<std::uintptr_t>(a) == reinterpret_cast<std::uintptr_t>(b);
//     }
// };


// struct CompareBySize {
//     bool operator()(const Block* a, const Block* b) const {
//         if (a->size != b->size) return a->size < b->size;
//         // tie-breaker by address (stable)
//         return std::uintptr_t(a->addr) < std::uintptr_t(b->addr);
//     }
// };


// struct StreamCache {
//     std::set<Block*> small_pool; // Blocks <1MB
//     std::set<Block*> large_pool; // Blocks >=1MB
// };


// struct StreamInternal {
//     // pools are sorted by size for best-fit
//     std::multiset<Block*, CompareBySize> small_pool; //allows to store multiple duplicates.(bins or blocks of the same size)
//     std::multiset<Block*, CompareBySize> large_pool;

//     // keep track of all blocks created for this stream (owns memory)
//     std::vector<Block*> all_blocks;
// };



// class CachingAllocator {
// public:
//     CachingAllocator();
//     ~CachingAllocator();

//     // Allocate memory for FSDP parameter
//     Block* allocateMemory(size_t size, cudaStream_t stream); 

//     // Free memory (return to cache, not GPU)
//     void freeMemory(Block* block);

//     // Force allocate memory from GPU
//     Block* requestMemory(size_t size, PoolType pool_type, cudaStream_t stream); //

//     // Clear cached memory (frees GPU memory)
//     void emptyCache(Device device = Device::CPU); 

//     // Memory statistics
//     size_t memoryAllocated() const;
//     size_t memoryFree() const;
//     void printStats() const;
//     Block* allocNewBlock(size_t size, cudaStream_t stream, PoolType pool_type=PoolType::SMALL);

//     // Utilities
//     size_t roundMemorySize(size_t size) const; //
//     bool isAllocated(Block* block) const; //

// private:
//     mutable std::mutex mutex_;  // Thread-safe operations
//     std::unordered_map<cudaStream_t, StreamInternal> stream_to_cache_;
//     std::atomic<size_t> total_allocated_;
//     std::atomic<size_t> total_free_;

//     // Find best-fit block in pool
//     Block* findBestFit(std::multiset<Block*, CompareBySize>& pool, size_t size);

//     // Split block if requested size < block size
//     Block* maybeSplitBlock(std::multiset<Block*, CompareBySize>& pool, Block* block, size_t requested_size);

//     // Merge adjacent free blocks
//     void mergeBlock(Block* block);

//     // Allocate new block from GPU
//     // Block* allocNewBlock(size_t size, cudaStream_t stream, PoolType pool_type=PoolType::SMALL);

//     // Determine pool type
//     PoolType selectPoolType(size_t size) const;
// };


#pragma once
#include <cstddef>
#include <unordered_map>
#include <set>
#include <mutex>
#include <atomic>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/*
 * FSDP-style Stream-aware GPU Caching Allocator
 * ---------------------------------------------
 * - Best-fit with splitting/merging
 * - Stream isolation
 * - Thread-safe
 * - Memory rounding and pooling
 * - Detailed statistics
 */

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
