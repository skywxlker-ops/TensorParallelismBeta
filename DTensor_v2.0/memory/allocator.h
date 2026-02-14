#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <iostream>

class MemoryAllocator {
public:
    virtual void* allocate(size_t size, cudaStream_t stream = 0) = 0;
    virtual void free(void* ptr, cudaStream_t stream = 0) = 0;
    virtual void emptyCache() = 0;
    virtual void printStats() const = 0;
    virtual ~MemoryAllocator() = default;
};

extern std::unique_ptr<MemoryAllocator> global_allocator;
