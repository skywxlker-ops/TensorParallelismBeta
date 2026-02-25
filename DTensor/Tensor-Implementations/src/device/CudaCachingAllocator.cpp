#include "device/CachingCudaAllocator.h"
#include "device/SizeClass.h"
#include "device/AllocationTracker.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <algorithm>

namespace OwnTensor
{
    CachingCUDAAllocator& CachingCUDAAllocator::instance()
    {
        static CachingCUDAAllocator allocator;
        return allocator;
    }

    CachingCUDAAllocator::CachingCUDAAllocator()
    {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        device_pools_.resize(device_count);
    }

    CachingCUDAAllocator::~CachingCUDAAllocator()
    {
        empty_cache();
    }

    void* CachingCUDAAllocator::allocate(size_t bytes)
    {
        cudaStream_t stream = cuda::getCurrentStream();
        return allocate(bytes, stream);
    }

    void CachingCUDAAllocator::trim_to(size_t target_bytes)
    {
        for (DevicePools& pools : device_pools_)
        {
            std::lock_guard<std::mutex> lock(*pools.mtx);
            trim_pool(pools.small_pool, target_bytes / 2);
            trim_pool(pools.large_pool, target_bytes / 2);
        }
    }

    void CachingCUDAAllocator::trim_pool(BlockPool& pool, size_t target_bytes) {
    // ASSUMES LOCK IS HELD

    // Use a while loop with a safe forward-iterator check
    while (pool.total_cached > target_bytes && !pool.free_blocks.empty()) {
        
        // Always take the largest block to satisfy OOM quickly
        // In std::set, the largest element is at the end
        auto it = std::prev(pool.free_blocks.end());
        Block* block = *it;

        // CRITICAL: Update pointers and stats BEFORE the slow cudaFree
        pool.total_cached -= block->size;
        // pool.allocated_blocks.erase(block->ptr); // Not in allocated blocks if it's in free_blocks
        pool.free_blocks.erase(it);

        // LOCK #2: Stats update (must be very brief)
        {
            std::lock_guard<std::mutex> s_lock(stats_mutex_);
            total_frees_++;
        }

        // PHYSICAL FREE
        // We do this while still holding pool.mtx to prevent 
        // another thread from getting the same pointer from the driver
        cudaStreamSynchronize(block->stream);
        cudaFree(block->ptr); 
        
        assert(block != nullptr);
        delete block;
    }
}

    void* CachingCUDAAllocator::allocate(size_t bytes, cudaStream_t stream)
    {
        if (bytes == 0) return nullptr;

        int device;
        cudaGetDevice(&device);

        size_t alloc_size = SizeClass::round_size(bytes);

        BlockPool& preferred_pool = get_pool(alloc_size, device);
        BlockPool* actual_pool_ptr = &preferred_pool;

        Block* block = nullptr;
        {
            std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);
            
            // Try preferred pool first
            block = preferred_pool.find_free_block(alloc_size, stream);
            
            // If missed in small pool, try large pool as fallback
            if (!block && SizeClass::is_small(alloc_size)) {
                block = device_pools_[device].large_pool.find_free_block(alloc_size, stream);
                if (block) {
                    // Pull from large pool!
                    actual_pool_ptr = &device_pools_[device].large_pool;
                }
            }

            if (block)
            {
                {
                    std::lock_guard<std::mutex> s_lock(stats_mutex_);
                    cache_hits_ += 1;
                }
                actual_pool_ptr->total_cached -= block->size;

                if (block->size >= alloc_size + SizeClass::kSmallSize)
                {
                    block = try_split(block, alloc_size);
                }
            }
        }

        if (block) {
            ensure_stream_safety(block, stream);
        }

        // Cache miss - need fresh CUDA allocation
        if (!block) {
            cache_misses_++;
            block = cuda_alloc(alloc_size, device, stream);
            if (!block) {
                // Log to system stderr directly to bypass potential stream hangs
                fprintf(stderr, "[ALLOCATOR] Hard OOM. Trimming cache...\n");
                num_ooms_++;
                
                this->trim_to(0); 
                
                block = cuda_alloc(alloc_size, device, stream);
                if (!block) {
                    fprintf(stderr, "[ALLOCATOR] FATAL: Still OOM after trim.\n");
                    throw std::runtime_error("CUDA OOM");
                }
                num_alloc_retries_++;
            }
        }


        block->allocated = true;
        block->req_size = bytes;
        block->stream = stream;

        {
            std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);
            actual_pool_ptr->allocated_blocks[block->ptr] = block;
        }

        AllocationTracker::instance().on_alloc(block->ptr, bytes, device);

        total_allocs_++;
        return block->ptr;
    }

    void CachingCUDAAllocator::deallocate(void* ptr)
    {
        if (!ptr) return;

        int device;
        cudaGetDevice(&device);

        Block* block = nullptr;
        BlockPool* pool_ptr = nullptr;

        {
            std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);
            for (BlockPool* pool :
                { &device_pools_[device].small_pool, &device_pools_[device].large_pool })
            {
                auto it = pool->allocated_blocks.find(ptr);
                if (it != pool->allocated_blocks.end())
                {
                    block = it->second;
                    pool_ptr = pool;
                    pool->allocated_blocks.erase(it);
                    break;
                }
            }

            if (!block)
            {
                std::cerr << "Warning: deallocate called on unknown pointer" << std::endl;
                return;
            }

            AllocationTracker::instance().on_free(ptr, device);

            block->allocated = false;
            block = pool_ptr->try_block_merge(block);

            BlockPool& final_pool = get_pool(block->size, device);

            final_pool.free_blocks.insert(block);
            final_pool.total_cached += block->size;
        }

        total_frees_++;
    }

    Block* CachingCUDAAllocator::cuda_alloc(size_t size, int device, cudaStream_t stream)
    {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocAsync(&ptr, size, stream);
        if (err != cudaSuccess || !ptr) {
            fprintf(stderr, "[ALLOCATOR] cudaMallocAsync(size=%zu) failed: %s\n", size, cudaGetErrorString(err));
            return nullptr;
        }

        Block* block = new Block(ptr, size, device, stream);

        BlockPool& pool = get_pool(size, device);
        {
            std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);
            pool.total_allocated += size;
            pool.peak_allocated = std::max(pool.peak_allocated, pool.total_allocated);
        }
        return block;
    }

    void CachingCUDAAllocator::cuda_free(Block* block)
    {
        cudaFreeAsync(block->ptr, block->stream);
        BlockPool& pool = get_pool(block->size, block->device_id);
        {
            std::lock_guard<std::mutex> lock(*device_pools_[block->device_id].mtx);
            pool.total_allocated -= block->size;
        }

        delete block;
    }

    void CachingCUDAAllocator::cuda_free_locked(Block* block)
    {
        // ASSUMES LOCK IS HELD
        if (block->prev == nullptr && block->next == nullptr) {
            cudaFreeAsync(block->ptr, block->stream);
            BlockPool& pool = get_pool(block->size, block->device_id);
            pool.total_allocated -= block->size;
        }
        delete block;
    }

    BlockPool& CachingCUDAAllocator::get_pool(size_t size, int device)
    {
        if (SizeClass::is_small(size)) {
            return device_pools_[device].small_pool;
        }
        else {
            return device_pools_[device].large_pool;
        }
    }

    void CachingCUDAAllocator::empty_cache()
    {
        // std::vector<Block*> blocks_to_free;
        // for (DevicePools& dev_pools : device_pools_)
        // {
        //      // Physical layout and pools are protected by device lock
        //      std::lock_guard<std::mutex> lock(*dev_pools.mtx);
        //      for (BlockPool* pool : { &dev_pools.small_pool, &dev_pools.large_pool })
        //      {
        //         for (Block* block : pool->free_blocks)
        //         {
        //             blocks_to_free.push_back(block);
        //         }
        //         pool->free_blocks.clear();
        //         pool->total_cached = 0;
        //      }
        // }

        // // You told me to add
        // // std::sort(blocks_to_free.begin(), blocks_to_free.end());
        // // blocks_to_free.erase(std::unique(blocks_to_free.begin(), blocks_to_free.end()), blocks_to_free.end());

        // for (Block* block : blocks_to_free)
        // {
        //     // we already cleared the pools' total_cached and free_blocks lists
        //     // but we need to update total_allocated and call Physical Free
        //     if (!block || !block->ptr) continue;
        //     cuda_free_locked(block);
        //     block->ptr = nullptr;
        // }
        
        // cudaDeviceSynchronize();

        std::set<Block*> unique_blocks_to_free;

        for (DevicePools& dev_pools : device_pools_) {
            std::lock_guard<std::mutex> lock(*dev_pools.mtx);
            for (BlockPool* pool : { &dev_pools.small_pool, &dev_pools.large_pool}) {
                for (Block* block : pool->free_blocks) {
                    unique_blocks_to_free.insert(block);
                }
                pool->free_blocks.clear();
                pool->total_cached = 0;
            }
        }

        for (Block* block : unique_blocks_to_free) {
            cuda_free_locked(block);
        }
        cudaDeviceSynchronize();
    }

    void CachingCUDAAllocator::ensure_stream_safety(Block* block, cudaStream_t target_stream)
    {
        if (block->stream == target_stream)
        {
            return;
        }

        if (block->stream == 0 || target_stream == 0)
        {
            cudaStreamSynchronize(block->stream);
        }
        else
        {
            cudaEvent_t event;
            cudaEventCreate(&event);
            cudaEventRecord(event, block->stream);
            cudaStreamWaitEvent(target_stream, event, 0);
            cudaEventDestroy(event);
        }

        block->stream = target_stream;
    }

    Block* CachingCUDAAllocator::try_split(Block* block, size_t size)
    {
        size_t remaining = block->size - size;

        if (remaining < SizeClass::kSmallSize)
        {
            return block; // too small to split }
        }
        
        num_splits_++;  // Track split operation
        
        void* new_ptr = static_cast<char*>(block->ptr) + size;
        Block* new_block = new Block(new_ptr, remaining, block->device_id, block->stream);
        new_block->is_split = true;

        block->size = size;
        block->is_split = true;

        new_block->prev = block;
        new_block->next = block->next;
        if (block->next)
        {
            block->next->prev = new_block;
        }
        block->next = new_block;


        BlockPool& pool = get_pool(remaining, block->device_id);
        // Already holding device lock from allocate()
        pool.free_blocks.insert(new_block);
        pool.total_cached += remaining;
        
        return block;
    }

    Block* BlockPool::find_free_block(size_t size, cudaStream_t /*stream*/)
    {
        Block search_key(nullptr, size, 0, nullptr);
        auto it = free_blocks.lower_bound(&search_key);

        if (it != free_blocks.end())
        {
            Block* block = *it;
            free_blocks.erase(it);
            return block;
        }
        return nullptr;
    }

    void BlockPool::return_block(Block* block)
    {
        free_blocks.insert(block);
        total_cached += block->size; 
    }

    Block* BlockPool::try_block_merge(Block* block)
    {
        // This is called from CachingCUDAAllocator with the device lock held.
        CachingCUDAAllocator& alloc = CachingCUDAAllocator::instance();

        auto remove_block = [&](Block* b) {
            bool removed = false;
            for (BlockPool* p : { &alloc.device_pools_[b->device_id].small_pool, 
                                 &alloc.device_pools_[b->device_id].large_pool }) {
                auto it = p->free_blocks.find(b);
                if (it != p->free_blocks.end()) {
                    p->free_blocks.erase(it);
                    p->total_cached -= b->size;
                    removed = true;
                    break;
                }
            }
            return removed;
        };

        if (block->prev && !block->prev->allocated)
        {
            Block* prev = block->prev;
            BlockPool& prev_pool = alloc.get_pool(prev->size, prev->device_id);
            
            prev_pool.free_blocks.erase(prev);
            prev_pool.total_cached -= prev->size;
            
            prev->size += block->size;
            prev->next = block->next;
            if (block->next)
            {
                block->next->prev = prev;
            }

            delete block;
            block = prev;
            alloc.num_merges_++;  // Track merge operation
        }

        if (block->next && !block->next->allocated)
        {
            Block* next = block->next;
            if (remove_block(next)) {
                block->size += next->size;
                block->next = next->next;
                if (next->next) {
                    next->next->prev = block;
                }
                delete next;
                alloc.num_merges_++;  // Track merge operation
            }

        }

        assert(block != nullptr);
        return block;
    }

    CachingCUDAAllocator::MemoryStats CachingCUDAAllocator::get_stats(int device) const
    {
        MemoryStats stats = {};

        auto add_pool_stats = [&](const BlockPool& pool, bool is_small) {
            size_t pool_active = pool.total_allocated - pool.total_cached;
            
            stats.active_current += pool_active;
            stats.allocated_current += pool.total_allocated;
            stats.reserved_current += pool.total_cached;
            stats.allocated_peak = std::max(stats.allocated_peak, pool.peak_allocated);
            
            if (is_small) {
                stats.small_pool_allocated += pool.total_allocated;
                stats.small_pool_cached += pool.total_cached;
            } else {
                stats.large_pool_allocated += pool.total_allocated;
                stats.large_pool_cached += pool.total_cached;
            }
        };

        if (device < 0)
        {
            for (const DevicePools& dev_pools : device_pools_)
            {
                std::lock_guard<std::mutex> lock(*dev_pools.mtx);
                add_pool_stats(dev_pools.small_pool, true);
                add_pool_stats(dev_pools.large_pool, false);
            }
        }
        else
        {
            if (device < (int)device_pools_.size()) {
                const DevicePools& dev_pools = device_pools_[device];
                std::lock_guard<std::mutex> lock(*dev_pools.mtx);
                add_pool_stats(dev_pools.small_pool, true);
                add_pool_stats(dev_pools.large_pool, false);
            }
        }

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats.num_allocs = total_allocs_;
            stats.num_frees = total_frees_;
            stats.num_cache_hits = cache_hits_;
            stats.num_cache_misses = cache_misses_;
            stats.num_splits = num_splits_;
            stats.num_merges = num_merges_;
            stats.num_ooms = num_ooms_;
            stats.num_alloc_retries = num_alloc_retries_;
            stats.active_peak = peak_active_;
            stats.reserved_peak = peak_reserved_;
        }
        
        // Legacy compatibility
        stats.allocated = stats.active_current;
        stats.cached = stats.reserved_current;
        stats.peak = stats.allocated_peak;
        
        return stats;
    }

    void CachingCUDAAllocator::print_memory_summary() const
    {
        MemoryStats stats = get_stats();
        
        auto mb = [](size_t bytes) { return bytes / 1024.0 / 1024.0; };

        std::cerr << "\n==================== CUDA Caching Allocator Stats ====================\n";
        
        std::cerr << "\n--- Memory Usage ---\n";
        std::cerr << "  Active (in use):     " << mb(stats.active_current) << " MB (peak: " << mb(stats.active_peak) << " MB)\n";
        std::cerr << "  Allocated (CUDA):    " << mb(stats.allocated_current) << " MB (peak: " << mb(stats.allocated_peak) << " MB)\n";
        std::cerr << "  Reserved (cached):   " << mb(stats.reserved_current) << " MB (peak: " << mb(stats.reserved_peak) << " MB)\n";
        
        std::cerr << "\n--- Pool Breakdown ---\n";
        std::cerr << "  Small pool:          " << mb(stats.small_pool_allocated) << " MB allocated, " 
                  << mb(stats.small_pool_cached) << " MB cached\n";
        std::cerr << "  Large pool:          " << mb(stats.large_pool_allocated) << " MB allocated, " 
                  << mb(stats.large_pool_cached) << " MB cached\n";
        
        std::cerr << "\n--- Allocation Stats ---\n";
        std::cerr << "  Total allocations:   " << stats.num_allocs << "\n";
        std::cerr << "  Total frees:         " << stats.num_frees << "\n";
        std::cerr << "  Cache hits:          " << stats.num_cache_hits << " (" << stats.cache_hit_rate() << "%)\n";
        std::cerr << "  Cache misses:        " << stats.num_cache_misses << "\n";
        
        std::cerr << "\n--- Block Operations ---\n";
        std::cerr << "  Block splits:        " << stats.num_splits << "\n";
        std::cerr << "  Block merges:        " << stats.num_merges << "\n";
        
        std::cerr << "\n--- OOM Recovery ---\n";
        std::cerr << "  OOM events:          " << stats.num_ooms << "\n";
        std::cerr << "  Successful retries:  " << stats.num_alloc_retries << "\n";
        
        std::cerr << "\n--- Derived Metrics ---\n";
        std::cerr << "  Fragmentation:       " << stats.fragmentation_ratio() << "%\n";
        
        std::cerr << "=======================================================================\n\n";
    }

}

