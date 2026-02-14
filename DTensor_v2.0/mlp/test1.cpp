#include<iostream>
#include <cuda_runtime.h>

int main(){
    size_t size=1024*1024*200;
    void* ptr = nullptr;
    cudaMalloc(&ptr,size);
    std::cout<<"cudaMalloc success"<<std::endl;
    std::cin.get();
    cudaFree(ptr);
    std::cout<<"cudaFree success"<<std::endl;
    void* ptr1=nullptr;
    cudaHostAlloc(&ptr1,size,cudaHostAllocPortable);
    std::cout<<"cudaHostAlloc success"<<std::endl;
    std::cin.get();
    cudaFreeHost(ptr1);
    std::cout<<"cudaFreeHost success"<<std::endl;
    std::cin.get();
    return 0;
}
