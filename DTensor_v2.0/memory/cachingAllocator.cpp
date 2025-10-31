// // // cachingAllocator.cpp
// // #include "cachingAllocator.hpp"

// // #include <algorithm>
// // #include <cassert>
// // #include <cstring>
// // #include <iostream>
// // #include <stdexcept>
// // #include <sstream>

// // // -----------------------------
// // // Helpers & comparators
// // // -----------------------------




// // CachingAllocator::CachingAllocator()
// //         : total_allocated_(0), total_free_(0) {
// //     }

// // CachingAllocator::~CachingAllocator() {
// //     try {
// //         emptyCache(Device::CUDA);
// //     } catch (...) {
// //         // Don't throw from destructor
// //     }
// // }


// // size_t CachingAllocator::roundMemorySize(size_t size) const {
// //     if (size == 0) return 0;
// //     if (size < ONE_MB) {
// //         // round up to ALIGNMENT (512) for small sizes
// //         size_t rem = size % ALIGNMENT;
// //         if (rem == 0) return size;
// //         return size + (ALIGNMENT - rem);
// //     } else {
// //         // round up to 1MB for large sizes (coarser granularity)
// //         size_t rem = size % ONE_MB;
// //         if (rem == 0) return size;
// //         return size + (ONE_MB - rem);
// //     }
// // }

// // PoolType CachingAllocator::selectPoolType(size_t size) const {
// //     return (size < ONE_MB) ? PoolType::SMALL : PoolType::LARGE;
// // }

// // /*
// //  * Find the best-fit block in the pool (smallest block >= requested size).
// //  * Using multiset ordered by size — we lower_bound on a fake Block with requested size.
// //  */

// // bool isround(size_t size){
// //     return size>=ONE_MB?size%ONE_MB==0:size%ALIGNMENT==0;
// // }


// // Block* CachingAllocator::findBestFit(std::multiset<Block*, CompareBySize>& pool, size_t size) {
// //     if (pool.empty()) return nullptr;
// //     Block fake(nullptr, size, PoolType::SMALL, 0);
// //     auto it = pool.lower_bound(&fake);
// //     if (it == pool.end()) return nullptr;
// //     return *it;
// // }

// // Block* CachingAllocator::maybeSplitBlock(std::multiset<Block*, CompareBySize>& pool, Block* block, size_t requested_size) {
// //     assert(block); 
// //     if (block->size < requested_size) return nullptr; // shouldn't happen

// //     size_t remaining = block->size - roundMemorySize(requested_size);
// //     if (remaining < MIN_SPLIT || !isround(remaining)) {
// //         // Not worth splitting: allocate whole block
// //         block->active = true;
// //         return block;
// //     }
 
// //     // Split: original block becomes the allocated part; create remainder block
// //     void* allocated_addr = block->addr;
// //     void* remainder_addr = reinterpret_cast<void*>(reinterpret_cast<char*>(block->addr) + requested_size);

// //     // modify existing block to be the allocated portion
// //     block->size = requested_size;
// //     block->active = true;

// //     // create remainder Block
// //     Block* rem = new Block(remainder_addr, remaining, block->pool_type, block->stream);
// //     rem->active = false;

// //     // link prev/next for possible future merging; keep them within same allocation context
// //     rem->prev = block;
// //     rem->next = block->next;
// //     if (block->next) block->next->prev = rem;
// //     block->next = rem;

// //     // Add remainder into the stream's pool (we must know which pool container — caller will handle)
// //     return block;
// // }

// // /*
// //  * Allocate a new block from GPU using cudaMalloc. On success, store Block in the stream's all_blocks.
// //  */
// // Block* CachingAllocator::allocNewBlock(size_t size, cudaStream_t stream, PoolType pool_type) {
// //     void* dev_ptr = nullptr;
// //     size_t sizeRounded=roundMemorySize(size);
// //     cudaError_t err = cudaMalloc(&dev_ptr, sizeRounded);
// //     if (err != cudaSuccess) {
// //         std::ostringstream os;
// //         os << "cudaMalloc failed (" << cudaGetErrorString(err) << ") for size " << sizeRounded;
// //         throw std::runtime_error(os.str());
// //     }

// //     Block* b = new Block(dev_ptr, sizeRounded, pool_type, stream);

// //     // record ownership in stream_to_cache_
// //     {
// //         // lock briefly to push into all_blocks
// //         std::lock_guard<std::mutex> lk(mutex_);
// //         // find or create internal stream entry
// //         auto it = stream_to_cache_.find(stream);
// //         if (it == stream_to_cache_.end()) {
// //             StreamInternal si;
// //             si.all_blocks.push_back(b);
// //             stream_to_cache_.emplace(stream, std::move(si));
// //         } else {
// //             it->second.all_blocks.push_back(b);
// //         }
// //     }

// //     total_allocated_.fetch_add(sizeRounded, std::memory_order_relaxed);
// //     return b;
// // }

// // Block* CachingAllocator::requestMemory(size_t size, PoolType pool_type, cudaStream_t stream) {
// //     size_t rsize = roundMemorySize(size);
// //     std::lock_guard<std::mutex> lk(mutex_);
// //     return allocNewBlock(rsize, stream, pool_type);
// // }
// // size_t CachingAllocator::memoryAllocated() const {
// //     return total_allocated_.load(std::memory_order_relaxed);
// // }
// // size_t CachingAllocator::memoryFree() const {
// //     return total_free_.load(std::memory_order_relaxed);
// // }


// // // Block* CachingAllocator::allocateMemory(size_t size, cudaStream_t stream) {
// // //     if (size == 0) return nullptr;

// // //     size_t rsize = roundMemorySize(size);
// // //     PoolType pool_type = selectPoolType(rsize);

// // //     std::lock_guard<std::mutex> lk(mutex_);

// // //     // ensure stream entry exists
// // //     auto sit = stream_to_cache_internal_.find(stream);
// // //     if (sit == stream_to_cache_internal_.end()) {
// // //         stream_to_cache_internal_.emplace(stream, StreamInternal{});
// // //         sit = stream_to_cache_internal_.find(stream);
// // //     }
// // //     StreamInternal &internal = sit->second;

// // //     std::multiset<Block*, CompareBySize> &pool =
// // //         (pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;

// // //     Block* candidate = findBestFit(pool, rsize);
// // //     if (candidate) {
// // //         // remove candidate from pool
// // //         pool.erase(pool.find(candidate));

// // //         // if splitting is beneficial
// // //         size_t orig_size = candidate->size;
// // //         if (orig_size > rsize && (orig_size - rsize) >= MIN_SPLIT) {
// // //             // perform split: candidate becomes allocated portion
// // //             void* allocated_addr = candidate->addr;
// // //             void* rem_addr = reinterpret_cast<void*>(reinterpret_cast<char*>(candidate->addr) + rsize);

// // //             // create remainder block
// // //             Block* remainder = new Block(rem_addr, orig_size - rsize, candidate->pool_type, candidate->stream);
// // //             remainder->active = false;

// // //             // link
// // //             remainder->prev = candidate;
// // //             remainder->next = candidate->next;
// // //             if (candidate->next) candidate->next->prev = remainder;
// // //             candidate->next = remainder;

// // //             // update candidate
// // //             candidate->size = rsize;
// // //             candidate->active = true;

// // //             // insert remainder into pool
// // //             pool.insert(remainder);

// // //             // ensure remainder is recorded in all_blocks list
// // //             internal.all_blocks.push_back(remainder);
// // //         } else {
// // //             candidate->active = true;
// // //         }

// // //         // Update counters
// // //         total_free_.fetch_sub(candidate->size, std::memory_order_relaxed);

// // //         return candidate;
// // //     }

// // //     // Cache miss -> allocate new block from GPU
// // //     try {
// // //         Block* nb = allocNewBlock(rsize, pool_type, stream);
// // //         nb->active = true;
// // //         // ensure new block is recorded (allocNewBlock added to all_blocks)
// // //         return nb;
// // //     } catch (...) {
// // //         // allocation failed: rethrow (caller can handle)
// // //         throw;
// // //     }
// // // }

// // // /*
// // //  * freeMemory:
// // //  * - marks block free
// // //  * - attempts to merge with adjacent free blocks (if they exist and are from same stream)
// // //  * - inserts into appropriate pool
// // //  */
// // // void CachingAllocator::freeMemory(Block* block) {
// // //     if (!block) return;

// // //     std::lock_guard<std::mutex> lk(mutex_);

// // //     if (!block->active) {
// // //         // double free — warn and ignore
// // //         std::cerr << "[CachingAllocator] warning: double free or freeing an already free block at " << block->addr << '\n';
// // //         return;
// // //     }

// // //     block->active = false;
// // //     total_free_.fetch_add(block->size, std::memory_order_relaxed);

// // //     // find the stream cache
// // //     auto sit = stream_to_cache_internal_.find(block->stream);
// // //     if (sit == stream_to_cache_internal_.end()) {
// // //         // If stream not found, create entry
// // //         stream_to_cache_internal_.emplace(block->stream, StreamInternal{});
// // //         sit = stream_to_cache_internal_.find(block->stream);
// // //     }
// // //     StreamInternal &internal = sit->second;
// // //     auto &pool = (block->pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;

// // //     // Attempt merge with previous block
// // //     if (block->prev && !block->prev->active && block->prev->stream == block->stream) {
// // //         // Remove prev from its pool (if it was present)
// // //         auto &prev_pool = (block->prev->pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;
// // //         auto pit = prev_pool.find(block->prev);
// // //         if (pit != prev_pool.end()) prev_pool.erase(pit);

// // //         // merge prev <- block
// // //         block->prev->size += block->size;
// // //         block->prev->next = block->next;
// // //         if (block->next) block->next->prev = block->prev;

// // //         // delete block structure (we keep prev as the merged block)
// // //         // remove block from all_blocks vector (lazy removal)
// // //         // We won't call cudaFree here because merged block still refers to same base addr
// // //         // remove block pointer from internal.all_blocks
// // //         auto it = std::find(internal.all_blocks.begin(), internal.all_blocks.end(), block);
// // //         if (it != internal.all_blocks.end()) internal.all_blocks.erase(it);
// // //         delete block;
// // //         block = block->prev;
// // //     }

// // //     // Attempt merge with next block
// // //     if (block->next && !block->next->active && block->next->stream == block->stream) {
// // //         auto &next_pool = (block->next->pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;
// // //         auto nit = next_pool.find(block->next);
// // //         if (nit != next_pool.end()) next_pool.erase(nit);

// // //         // merge block <- next
// // //         block->size += block->next->size;
// // //         Block* nextb = block->next;
// // //         block->next = nextb->next;
// // //         if (nextb->next) nextb->next->prev = block;

// // //         // remove nextb from all_blocks
// // //         auto it = std::find(internal.all_blocks.begin(), internal.all_blocks.end(), nextb);
// // //         if (it != internal.all_blocks.end()) internal.all_blocks.erase(it);
// // //         delete nextb;
// // //     }

// // //     // Insert the (possibly merged) block into pool
// // //     pool.insert(block);
// // // }

// // /*
// //  * Force an allocation from GPU (explicit API)
// //  */


// // /*
// //  * emptyCache:
// //  * Frees all GPU memory we own (for the given device). If device==CPU, does nothing for GPU memory.
// //  * NOTE: This is destructive: all Block* pointers returned earlier become invalid.
// //  */
// // void CachingAllocator::emptyCache(Device device) {
// //     if (device == Device::CPU) {
// //         // nothing to free on CPU from this allocator
// //         return;
// //     }

// //     std::lock_guard<std::mutex> lk(mutex_);

// //     // For each stream, free all blocks' device memory
// //     for (auto &entry : stream_to_cache_) {
// //         StreamInternal &internal = entry.second;
// //         for (Block* b : internal.all_blocks) {
// //             if (b->addr) {
// //                 cudaError_t err = cudaFree(b->addr);
// //                 if (err != cudaSuccess) {
// //                     std::cerr << "[CachingAllocator] cudaFree failed for " << b->addr
// //                               << " : " << cudaGetErrorString(err) << '\n';
// //                     // continue trying to free others
// //                 } else {
// //                     total_allocated_.fetch_sub(b->size, std::memory_order_relaxed);
// //                 }
// //             }
// //             delete b;
// //         }
// //         internal.all_blocks.clear();
// //         internal.small_pool.clear();
// //         internal.large_pool.clear();
// //     }
// //     stream_to_cache_.clear();

// //     total_free_.store(0, std::memory_order_relaxed);
// // }

// // /*
// //  * memoryAllocated & memoryFree
// //  */


// // void CachingAllocator::printStats() const {
// //     std::lock_guard<std::mutex> lk(mutex_);
// //     std::cout << "CachingAllocator stats:\n";
// //     std::cout << "  Total allocated (bytes): " << memoryAllocated() << '\n';
// //     std::cout << "  Total free in caches (bytes): " << memoryFree() << '\n';
// //     std::cout << "  Streams cached: " << stream_to_cache_.size() << '\n';
// //     for (auto &entry : stream_to_cache_) {
// //         const StreamInternal &internal = entry.second; // const because this function is const
// //         size_t small_count = internal.small_pool.size();
// //         size_t large_count = internal.large_pool.size();

// //         std::cout << "    Stream [" << reinterpret_cast<std::uintptr_t>(entry.first) << "]: small_blocks=" 
// //                 << small_count
// //                 << ", large_blocks=" << large_count 
// //                 << ", total_blocks_owned=" << internal.all_blocks.size() << '\n';
// //     }
// // }

// // /*
// //  * Check if block currently allocated
// //  */
// // bool CachingAllocator::isAllocated(Block* block) const {
// //     if (!block) return false;
// //     return block->active;
// // }

// // // int main() {
// // //     CachingAllocator ca;
// // //     ca.printStats();
// // //     return 0;
// // // }


// #include "cachingAllocator.hpp"
// #include <cuda_runtime.h>
// #include <iostream>
// #include <algorithm>
// #include <mutex>

// #define MIN_SPLIT (1 << 20) // 1MB minimum split size

// CachingAllocator::CachingAllocator() : total_allocated_(0), total_free_(0) {}

// CachingAllocator::~CachingAllocator() {
//     std::lock_guard<std::mutex> lk(mutex_);
//     for (auto &kv : stream_to_cache_) {
//         for (Block* b : kv.second.all_blocks) {
//             if (b->addr) cudaFree(b->addr);
//             delete b;
//         }
//     }
//     stream_to_cache_.clear();
// }

// Block* CachingAllocator::allocateMemory(size_t size, cudaStream_t stream) {
//     std::lock_guard<std::mutex> lk(mutex_);
//     auto rounded = roundMemorySize(size);
//     PoolType pool = selectPoolType(rounded);
//     StreamCache& cache = stream_to_cache_[stream];

//     // Try to reuse a free block
//     for (auto it = cache.free_blocks.begin(); it != cache.free_blocks.end(); ++it) {
//         Block* block = *it;
//         if (block->size >= rounded && block->pool == pool) {
//             cache.free_blocks.erase(it);
//             cache.active_blocks.push_back(block);
//             total_free_ -= block->size;
//             return block;
//         }
//     }

//     // No reusable block found — allocate new
//     void* ptr = nullptr;
//     cudaError_t err = cudaMalloc(&ptr, rounded);
//     if (err != cudaSuccess) {
//         std::cerr << "[Allocator] cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
//         return nullptr;
//     }

//     Block* block = new Block(ptr, rounded, pool, stream);
//     cache.all_blocks.push_back(block);
//     cache.active_blocks.push_back(block);
//     total_allocated_ += rounded;

//     return block;
// }

// void CachingAllocator::freeMemory(Block* block) {
//     if (!block) return;
//     std::lock_guard<std::mutex> lk(mutex_);
//     StreamCache& cache = stream_to_cache_[block->stream];

//     // Remove from active list
//     auto it = std::find(cache.active_blocks.begin(), cache.active_blocks.end(), block);
//     if (it != cache.active_blocks.end()) cache.active_blocks.erase(it);

//     cache.free_blocks.push_back(block);
//     total_free_ += block->size;
// }

// void CachingAllocator::releaseAll() {
//     std::lock_guard<std::mutex> lk(mutex_);
//     for (auto &kv : stream_to_cache_) {
//         for (Block* b : kv.second.all_blocks) {
//             if (b->addr) cudaFree(b->addr);
//             delete b;
//         }
//     }
//     stream_to_cache_.clear();
//     total_allocated_ = 0;
//     total_free_ = 0;
// }

// size_t CachingAllocator::roundMemorySize(size_t size) const {
//     const size_t alignment = 512;
//     return ((size + alignment - 1) / alignment) * alignment;
// }

// CachingAllocator::PoolType CachingAllocator::selectPoolType(size_t size) {
//     return (size <= (1 << 22)) ? PoolType::SMALL : PoolType::LARGE;
// }

// void CachingAllocator::printStats() const {
//     std::lock_guard<std::mutex> lk(mutex_);
//     std::cout << "[Allocator Stats] Total Allocated: " << total_allocated_.load()
//               << " bytes | Total Free: " << total_free_.load() << " bytes" << std::endl;
// }


#include "cachingAllocator.hpp"
#include <algorithm>
#include <cassert>
#include <sstream>

CachingAllocator::CachingAllocator()
    : total_allocated_(0), total_free_(0) {}

CachingAllocator::~CachingAllocator() {
    try {
        emptyCache(Device::CUDA);
    } catch (...) {
        // destructor must not throw
    }
}

size_t CachingAllocator::roundMemorySize(size_t size) const {
    if (size == 0) return 0;
    if (size < ONE_MB) {
        size_t rem = size % ALIGNMENT;
        return rem ? size + (ALIGNMENT - rem) : size;
    } else {
        size_t rem = size % ONE_MB;
        return rem ? size + (ONE_MB - rem) : size;
    }
}

PoolType CachingAllocator::selectPoolType(size_t size) const {
    return (size < ONE_MB) ? PoolType::SMALL : PoolType::LARGE;
}

Block* CachingAllocator::allocNewBlock(size_t size, cudaStream_t stream, PoolType pool_type) {
    void* dev_ptr = nullptr;
    size_t rsize = roundMemorySize(size);
    cudaError_t err = cudaMalloc(&dev_ptr, rsize);
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << "cudaMalloc failed (" << cudaGetErrorString(err) << ") for size " << rsize;
        throw std::runtime_error(os.str());
    }

    Block* b = new Block(dev_ptr, rsize, pool_type, stream);
    auto& internal = stream_to_cache_[stream];
    internal.all_blocks.push_back(b);
    total_allocated_ += rsize;
    return b;
}

Block* CachingAllocator::findBestFit(std::multiset<Block*, CompareBySize>& pool, size_t size) {
    if (pool.empty()) return nullptr;
    Block fake(nullptr, size, PoolType::SMALL, 0);
    auto it = pool.lower_bound(&fake);
    return (it == pool.end()) ? nullptr : *it;
}

Block* CachingAllocator::allocateMemory(size_t size, cudaStream_t stream) {
    if (size == 0) return nullptr;

    size_t rsize = roundMemorySize(size);
    PoolType pool_type = selectPoolType(rsize);

    std::lock_guard<std::mutex> lk(mutex_);
    auto& internal = stream_to_cache_[stream];
    auto& pool = (pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;

    Block* candidate = findBestFit(pool, rsize);
    if (candidate) {
        pool.erase(pool.find(candidate));
        candidate->active = true;
        total_free_ -= candidate->size;

        // Optional split for large blocks
        if (candidate->size > rsize + MIN_SPLIT) {
            size_t rem_size = candidate->size - rsize;
            void* rem_addr = reinterpret_cast<void*>(reinterpret_cast<char*>(candidate->addr) + rsize);
            candidate->size = rsize;
            Block* remainder = new Block(rem_addr, rem_size, pool_type, stream);
            internal.all_blocks.push_back(remainder);
            pool.insert(remainder);
        }

        return candidate;
    }

    // No free block — allocate fresh
    Block* nb = allocNewBlock(rsize, stream, pool_type);
    nb->active = true;
    return nb;
}

void CachingAllocator::freeMemory(Block* block) {
    if (!block) return;

    std::lock_guard<std::mutex> lk(mutex_);
    if (!block->active) {
        std::cerr << "[Allocator Warning] Double free on block " << block->addr << std::endl;
        return;
    }

    block->active = false;
    total_free_ += block->size;

    auto it = stream_to_cache_.find(block->stream);
    if (it == stream_to_cache_.end()) return;
    StreamInternal& internal = it->second;
    auto& pool = (block->pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;

    mergeAdjacent(block, internal);
    pool.insert(block);
}

void CachingAllocator::mergeAdjacent(Block* block, StreamInternal& internal) {
    // Merge previous
    if (block->prev && !block->prev->active) {
        block->prev->size += block->size;
        block->prev->next = block->next;
        if (block->next) block->next->prev = block->prev;
        delete block;
        block = block->prev;
    }

    // Merge next
    if (block->next && !block->next->active) {
        block->size += block->next->size;
        Block* nxt = block->next;
        block->next = nxt->next;
        if (nxt->next) nxt->next->prev = block;
        delete nxt;
    }
}

void CachingAllocator::emptyCache(Device device) {
    if (device == Device::CPU) return;

    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& entry : stream_to_cache_) {
        StreamInternal& internal = entry.second;
        for (Block* b : internal.all_blocks) {
            if (b->addr) cudaFree(b->addr);
            delete b;
        }
        internal.all_blocks.clear();
        internal.small_pool.clear();
        internal.large_pool.clear();
    }
    stream_to_cache_.clear();
    total_allocated_ = 0;
    total_free_ = 0;
}

size_t CachingAllocator::memoryAllocated() const {
    return total_allocated_.load(std::memory_order_relaxed);
}

size_t CachingAllocator::memoryFree() const {
    return total_free_.load(std::memory_order_relaxed);
}

bool CachingAllocator::isAllocated(Block* block) const {
    return block && block->active;
}

void CachingAllocator::printStats() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::cout << "\n[GPU CachingAllocator Stats]\n";
    std::cout << "  Total Allocated: " << memoryAllocated() << " bytes\n";
    std::cout << "  Total Free:      " << memoryFree() << " bytes\n";
    std::cout << "  Streams Cached:  " << stream_to_cache_.size() << "\n";
    for (auto& kv : stream_to_cache_) {
        const StreamInternal& s = kv.second;
        std::cout << "    Stream[" << reinterpret_cast<std::uintptr_t>(kv.first)
                  << "]: small=" << s.small_pool.size()
                  << ", large=" << s.large_pool.size()
                  << ", total_blocks=" << s.all_blocks.size() << "\n";
    }
}
