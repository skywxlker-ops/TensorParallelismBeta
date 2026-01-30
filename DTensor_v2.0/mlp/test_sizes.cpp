#include <iostream>
#include <unparalleled/unparalleled.h>
#include "memory/cachingAllocator.hpp"

int main() {
    std::cout << "Block size: " << sizeof(OwnTensor::Block) << std::endl;
    std::cout << "CachingAllocator size: " << sizeof(OwnTensor::CachingAllocator) << std::endl;
    std::cout << "Mutex size: " << sizeof(std::mutex) << std::endl;
    std::cout << "UnorderedMap size: " << sizeof(std::unordered_map<cudaStream_t, OwnTensor::StreamInternal>) << std::endl;
    std::cout << "Atomic size: " << sizeof(std::atomic<size_t>) << std::endl;
    return 0;
}
