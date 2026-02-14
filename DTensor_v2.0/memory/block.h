#pragma once
#include <cuda_runtime.h>
#include <cstddef>

struct MemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    bool active = false;
    cudaStream_t stream = 0;

    MemoryBlock() = default;
    MemoryBlock(void* p, size_t s, cudaStream_t st = 0)
        : ptr(p), size(s), active(true), stream(st) {}
};
