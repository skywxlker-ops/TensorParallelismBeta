// cachingAllocator.cpp
#include "cachingAllocator.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sstream>

// -----------------------------
// Helpers & comparators
// -----------------------------




CachingAllocator::CachingAllocator()
        : total_allocated_(0), total_free_(0) {
    }

CachingAllocator::~CachingAllocator() {
    try {
        emptyCache(Device::CUDA);
    } catch (...) {
        // Don't throw from destructor
    }
}


size_t CachingAllocator::roundMemorySize(size_t size) const {
    if (size == 0) return 0;
    if (size < ONE_MB) {
        // round up to ALIGNMENT (512) for small sizes
        size_t rem = size % ALIGNMENT;
        if (rem == 0) return size;
        return size + (ALIGNMENT - rem);
    } else {
        // round up to 1MB for large sizes (coarser granularity)
        size_t rem = size % ONE_MB;
        if (rem == 0) return size;
        return size + (ONE_MB - rem);
    }
}

PoolType CachingAllocator::selectPoolType(size_t size) const {
    return (size < ONE_MB) ? PoolType::SMALL : PoolType::LARGE;
}

/*
 * Find the best-fit block in the pool (smallest block >= requested size).
 * Using multiset ordered by size — we lower_bound on a fake Block with requested size.
 */

bool isround(size_t size){
    return size>=ONE_MB?size%ONE_MB==0:size%ALIGNMENT==0;
}


Block* CachingAllocator::findBestFit(std::multiset<Block*, CompareBySize>& pool, size_t size) {
    if (pool.empty()) return nullptr;
    Block fake(nullptr, size, PoolType::SMALL, 0);
    auto it = pool.lower_bound(&fake);
    if (it == pool.end()) return nullptr;
    return *it;
}

Block* CachingAllocator::maybeSplitBlock(std::multiset<Block*, CompareBySize>& pool, Block* block, size_t requested_size) {
    assert(block); 
    if (block->size < requested_size) return nullptr; // shouldn't happen

    size_t remaining = block->size - roundMemorySize(requested_size);
    if (remaining < MIN_SPLIT || !isround(remaining)) {
        // Not worth splitting: allocate whole block
        block->active = true;
        return block;
    }
 
    // Split: original block becomes the allocated part; create remainder block
    void* allocated_addr = block->addr;
    void* remainder_addr = reinterpret_cast<void*>(reinterpret_cast<char*>(block->addr) + requested_size);

    // modify existing block to be the allocated portion
    block->size = requested_size;
    block->active = true;

    // create remainder Block
    Block* rem = new Block(remainder_addr, remaining, block->pool_type, block->stream);
    rem->active = false;

    // link prev/next for possible future merging; keep them within same allocation context
    rem->prev = block;
    rem->next = block->next;
    if (block->next) block->next->prev = rem;
    block->next = rem;

    // Add remainder into the stream's pool (we must know which pool container — caller will handle)
    return block;
}

/*
 * Allocate a new block from GPU using cudaMalloc. On success, store Block in the stream's all_blocks.
 */
Block* CachingAllocator::allocNewBlock(size_t size, cudaStream_t stream, PoolType pool_type) {
    void* dev_ptr = nullptr;
    size_t sizeRounded=roundMemorySize(size);
    cudaError_t err = cudaMalloc(&dev_ptr, sizeRounded);
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << "cudaMalloc failed (" << cudaGetErrorString(err) << ") for size " << sizeRounded;
        throw std::runtime_error(os.str());
    }

    Block* b = new Block(dev_ptr, sizeRounded, pool_type, stream);

    // record ownership in stream_to_cache_
    {
        // lock briefly to push into all_blocks
        std::lock_guard<std::mutex> lk(mutex_);
        // find or create internal stream entry
        auto it = stream_to_cache_.find(stream);
        if (it == stream_to_cache_.end()) {
            StreamInternal si;
            si.all_blocks.push_back(b);
            stream_to_cache_.emplace(stream, std::move(si));
        } else {
            it->second.all_blocks.push_back(b);
        }
    }

    total_allocated_.fetch_add(sizeRounded, std::memory_order_relaxed);
    return b;
}

Block* CachingAllocator::requestMemory(size_t size, PoolType pool_type, cudaStream_t stream) {
    size_t rsize = roundMemorySize(size);
    std::lock_guard<std::mutex> lk(mutex_);
    return allocNewBlock(rsize, stream, pool_type);
}
size_t CachingAllocator::memoryAllocated() const {
    return total_allocated_.load(std::memory_order_relaxed);
}
size_t CachingAllocator::memoryFree() const {
    return total_free_.load(std::memory_order_relaxed);
}


// Block* CachingAllocator::allocateMemory(size_t size, cudaStream_t stream) {
//     if (size == 0) return nullptr;

//     size_t rsize = roundMemorySize(size);
//     PoolType pool_type = selectPoolType(rsize);

//     std::lock_guard<std::mutex> lk(mutex_);

//     // ensure stream entry exists
//     auto sit = stream_to_cache_internal_.find(stream);
//     if (sit == stream_to_cache_internal_.end()) {
//         stream_to_cache_internal_.emplace(stream, StreamInternal{});
//         sit = stream_to_cache_internal_.find(stream);
//     }
//     StreamInternal &internal = sit->second;

//     std::multiset<Block*, CompareBySize> &pool =
//         (pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;

//     Block* candidate = findBestFit(pool, rsize);
//     if (candidate) {
//         // remove candidate from pool
//         pool.erase(pool.find(candidate));

//         // if splitting is beneficial
//         size_t orig_size = candidate->size;
//         if (orig_size > rsize && (orig_size - rsize) >= MIN_SPLIT) {
//             // perform split: candidate becomes allocated portion
//             void* allocated_addr = candidate->addr;
//             void* rem_addr = reinterpret_cast<void*>(reinterpret_cast<char*>(candidate->addr) + rsize);

//             // create remainder block
//             Block* remainder = new Block(rem_addr, orig_size - rsize, candidate->pool_type, candidate->stream);
//             remainder->active = false;

//             // link
//             remainder->prev = candidate;
//             remainder->next = candidate->next;
//             if (candidate->next) candidate->next->prev = remainder;
//             candidate->next = remainder;

//             // update candidate
//             candidate->size = rsize;
//             candidate->active = true;

//             // insert remainder into pool
//             pool.insert(remainder);

//             // ensure remainder is recorded in all_blocks list
//             internal.all_blocks.push_back(remainder);
//         } else {
//             candidate->active = true;
//         }

//         // Update counters
//         total_free_.fetch_sub(candidate->size, std::memory_order_relaxed);

//         return candidate;
//     }

//     // Cache miss -> allocate new block from GPU
//     try {
//         Block* nb = allocNewBlock(rsize, pool_type, stream);
//         nb->active = true;
//         // ensure new block is recorded (allocNewBlock added to all_blocks)
//         return nb;
//     } catch (...) {
//         // allocation failed: rethrow (caller can handle)
//         throw;
//     }
// }

// /*
//  * freeMemory:
//  * - marks block free
//  * - attempts to merge with adjacent free blocks (if they exist and are from same stream)
//  * - inserts into appropriate pool
//  */
// void CachingAllocator::freeMemory(Block* block) {
//     if (!block) return;

//     std::lock_guard<std::mutex> lk(mutex_);

//     if (!block->active) {
//         // double free — warn and ignore
//         std::cerr << "[CachingAllocator] warning: double free or freeing an already free block at " << block->addr << '\n';
//         return;
//     }

//     block->active = false;
//     total_free_.fetch_add(block->size, std::memory_order_relaxed);

//     // find the stream cache
//     auto sit = stream_to_cache_internal_.find(block->stream);
//     if (sit == stream_to_cache_internal_.end()) {
//         // If stream not found, create entry
//         stream_to_cache_internal_.emplace(block->stream, StreamInternal{});
//         sit = stream_to_cache_internal_.find(block->stream);
//     }
//     StreamInternal &internal = sit->second;
//     auto &pool = (block->pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;

//     // Attempt merge with previous block
//     if (block->prev && !block->prev->active && block->prev->stream == block->stream) {
//         // Remove prev from its pool (if it was present)
//         auto &prev_pool = (block->prev->pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;
//         auto pit = prev_pool.find(block->prev);
//         if (pit != prev_pool.end()) prev_pool.erase(pit);

//         // merge prev <- block
//         block->prev->size += block->size;
//         block->prev->next = block->next;
//         if (block->next) block->next->prev = block->prev;

//         // delete block structure (we keep prev as the merged block)
//         // remove block from all_blocks vector (lazy removal)
//         // We won't call cudaFree here because merged block still refers to same base addr
//         // remove block pointer from internal.all_blocks
//         auto it = std::find(internal.all_blocks.begin(), internal.all_blocks.end(), block);
//         if (it != internal.all_blocks.end()) internal.all_blocks.erase(it);
//         delete block;
//         block = block->prev;
//     }

//     // Attempt merge with next block
//     if (block->next && !block->next->active && block->next->stream == block->stream) {
//         auto &next_pool = (block->next->pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;
//         auto nit = next_pool.find(block->next);
//         if (nit != next_pool.end()) next_pool.erase(nit);

//         // merge block <- next
//         block->size += block->next->size;
//         Block* nextb = block->next;
//         block->next = nextb->next;
//         if (nextb->next) nextb->next->prev = block;

//         // remove nextb from all_blocks
//         auto it = std::find(internal.all_blocks.begin(), internal.all_blocks.end(), nextb);
//         if (it != internal.all_blocks.end()) internal.all_blocks.erase(it);
//         delete nextb;
//     }

//     // Insert the (possibly merged) block into pool
//     pool.insert(block);
// }

/*
 * Force an allocation from GPU (explicit API)
 */


/*
 * emptyCache:
 * Frees all GPU memory we own (for the given device). If device==CPU, does nothing for GPU memory.
 * NOTE: This is destructive: all Block* pointers returned earlier become invalid.
 */
void CachingAllocator::emptyCache(Device device) {
    if (device == Device::CPU) {
        // nothing to free on CPU from this allocator
        return;
    }

    std::lock_guard<std::mutex> lk(mutex_);

    // For each stream, free all blocks' device memory
    for (auto &entry : stream_to_cache_) {
        StreamInternal &internal = entry.second;
        for (Block* b : internal.all_blocks) {
            if (b->addr) {
                cudaError_t err = cudaFree(b->addr);
                if (err != cudaSuccess) {
                    std::cerr << "[CachingAllocator] cudaFree failed for " << b->addr
                              << " : " << cudaGetErrorString(err) << '\n';
                    // continue trying to free others
                } else {
                    total_allocated_.fetch_sub(b->size, std::memory_order_relaxed);
                }
            }
            delete b;
        }
        internal.all_blocks.clear();
        internal.small_pool.clear();
        internal.large_pool.clear();
    }
    stream_to_cache_.clear();

    total_free_.store(0, std::memory_order_relaxed);
}

/*
 * memoryAllocated & memoryFree
 */


void CachingAllocator::printStats() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::cout << "CachingAllocator stats:\n";
    std::cout << "  Total allocated (bytes): " << memoryAllocated() << '\n';
    std::cout << "  Total free in caches (bytes): " << memoryFree() << '\n';
    std::cout << "  Streams cached: " << stream_to_cache_.size() << '\n';
    for (auto &entry : stream_to_cache_) {
        const StreamInternal &internal = entry.second; // const because this function is const
        size_t small_count = internal.small_pool.size();
        size_t large_count = internal.large_pool.size();

        std::cout << "    Stream [" << reinterpret_cast<std::uintptr_t>(entry.first) << "]: small_blocks=" 
                << small_count
                << ", large_blocks=" << large_count 
                << ", total_blocks_owned=" << internal.all_blocks.size() << '\n';
    }
}

/*
 * Check if block currently allocated
 */
bool CachingAllocator::isAllocated(Block* block) const {
    if (!block) return false;
    return block->active;
}

// int main() {
//     CachingAllocator ca;
//     ca.printStats();
//     return 0;
// }
