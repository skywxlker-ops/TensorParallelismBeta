#pragma once

#include "device/Allocator.h"
#include "device/Block.h"
#include "device/BlockPool.h"
#include <vector>
#include <cassert>
#include <memory>
#include <deque>

namespace OwnTensor
{
    class CachingCUDAAllocator : public Allocator {
        friend class BlockPool;
        public:
            static CachingCUDAAllocator& instance();

            void* allocate(size_t bytes) override;
            void deallocate(void* ptr) override;

            
            void* allocate(size_t bytes, cudaStream_t stream);

            void empty_cache();

            void trim_to(size_t target_bytes);
            void trim_pool(BlockPool& pool, size_t target_bytes);

            struct MemoryStats {
                // Active memory (currently in use by tensors)
                size_t active_current;       // Memory currently allocated to tensors
                size_t active_peak;          // Peak active memory
                
                // Allocated memory (requested from CUDA, may include cached blocks)
                size_t allocated_current;    // Total allocated from CUDA
                size_t allocated_peak;       // Peak allocated
                
                // Reserved/Cached memory (held in cache, not in use)
                size_t reserved_current;     // Memory in free pools (cached)
                size_t reserved_peak;        // Peak reserved
                
                // Allocation counters
                size_t num_allocs;           // Total allocation requests
                size_t num_frees;            // Total deallocation requests
                size_t num_cache_hits;       // Times satisfied from cache
                size_t num_cache_misses;     // Times required fresh CUDA allocation
                
                // Split block tracking
                size_t num_splits;           // Number of block splits performed
                size_t num_merges;           // Number of block merges performed
                size_t inactive_split_bytes; // Bytes in split but inactive blocks
                
                // OOM and retry tracking
                size_t num_ooms;             // Number of OOM events
                size_t num_alloc_retries;    // Successful allocations after OOM retry
                
                // Pool-specific stats
                size_t small_pool_allocated; // Bytes in small pool
                size_t large_pool_allocated; // Bytes in large pool
                size_t small_pool_cached;    // Cached bytes in small pool
                size_t large_pool_cached;    // Cached bytes in large pool
                
                // Derived metrics
                double cache_hit_rate() const {
                    return num_allocs > 0 ? 100.0 * num_cache_hits / num_allocs : 0.0;
                }
                
                double fragmentation_ratio() const {
                    return allocated_current > 0 
                        ? 100.0 * (allocated_current - active_current) / allocated_current 
                        : 0.0;
                }
                
                // Legacy compatibility aliases
                size_t allocated = 0;  // = active_current
                size_t cached = 0;     // = reserved_current  
                size_t peak = 0;       // = allocated_peak
            };

            MemoryStats get_stats(int device = -1) const;

            void print_memory_summary() const;
            BlockPool& get_pool(size_t size, int device);

        private:
            CachingCUDAAllocator();
            ~CachingCUDAAllocator();

            struct DevicePools {
                BlockPool small_pool;
                BlockPool large_pool;
                std::unique_ptr<std::mutex> mtx; // Global lock for both pools on this device
                
                DevicePools() : mtx(std::make_unique<std::mutex>()) {}
                DevicePools(DevicePools&& other) noexcept = default;
                DevicePools& operator=(DevicePools&& other) noexcept = default;
            };

            std::deque<DevicePools> device_pools_;

            Block* cuda_alloc(size_t size, int device, cudaStream_t stream);

            void cuda_free(Block* block);
            void cuda_free_locked(Block* block);

            Block* try_split(Block* block, size_t size);

            void ensure_stream_safety(Block* block, cudaStream_t target_stream);

            mutable std::mutex stats_mutex_;
            size_t total_allocs_ = 0;
            size_t total_frees_ = 0;
            size_t cache_hits_ = 0;
            size_t cache_misses_ = 0;
            size_t num_splits_ = 0;
            size_t num_merges_ = 0;
            size_t num_ooms_ = 0;
            size_t num_alloc_retries_ = 0;
            size_t peak_active_ = 0;
            size_t peak_allocated_ = 0;
            size_t peak_reserved_ = 0;
    };
}