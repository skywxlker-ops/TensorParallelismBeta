#pragma once

#include "device/Block.h"
#include <set>
#include <unordered_map>
#include <mutex>

namespace OwnTensor
{
    class BlockPool 
    {
        public:
        
            
            // Free Blocks ordered by size
            std::set<Block*, BlockSizeComparator> free_blocks;

        // All blocks by pointer for lookup 
            std:: unordered_map<void*, Block*> allocated_blocks;

            size_t total_allocated = 0;
            size_t total_cached = 0;
            size_t peak_allocated = 0;


            // to find best fit block at least size bytes
            Block* find_free_block(size_t size, cudaStream_t stream);

            // return block to pool
            void return_block(Block* block);

            // try to coalesce / merge with adjacent free blocks to make a large free Block
            Block* try_block_merge(Block* block);
    };
}