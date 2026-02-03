#include "cachingAllocator.hpp"
#include <algorithm>
#include <cassert>
#include <sstream>

CachingAllocator::CachingAllocator()
    : total_allocated_(0), total_free_(0) {}

CachingAllocator::~CachingAllocator() {
    try {
        emptyCache(AllocatorDevice::CUDA);
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

void CachingAllocator::emptyCache(AllocatorDevice device) {
    if (device == AllocatorDevice::CPU) return;

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
