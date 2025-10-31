#include "allocator.h"
#include <iostream>

int main() {
    std::cout << "Testing SimpleCachingAllocator...\n";

    void* p1 = global_allocator->allocate(1024 * 1024);   // 1 MB
    void* p2 = global_allocator->allocate(512 * 1024);    // 512 KB

    global_allocator->printStats();

    global_allocator->free(p1);
    global_allocator->printStats();

    global_allocator->emptyCache();
    global_allocator->printStats();

    return 0;
}
