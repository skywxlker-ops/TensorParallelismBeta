#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// ============================================================================
// RAII Wrapper for GPU Memory
// ============================================================================

template<typename T>
class GPUArray {
private:
    T* ptr_;
    size_t size_;

public:
    // Constructor: Acquires GPU memory
    GPUArray(size_t n) : ptr_(nullptr), size_(n) {
        cudaError_t err = cudaMalloc(&ptr_, n * sizeof(T));
        if (err != cudaSuccess || ptr_ == nullptr) {
            throw std::runtime_error(
                std::string("cudaMalloc failed: ") + cudaGetErrorString(err)
            );
        }
    }
    
    // Destructor: Releases GPU memory (RAII guarantee!)
    ~GPUArray() {
        if (ptr_) {
            cudaFree(ptr_);  // ✅ Always called when object destroyed
        }
    }
    
    // Delete copy constructor and assignment (no shallow copies)
    GPUArray(const GPUArray&) = delete;
    GPUArray& operator=(const GPUArray&) = delete;
    
    // Move constructor (transfer ownership)
    GPUArray(GPUArray&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    // Move assignment (transfer ownership)
    GPUArray& operator=(GPUArray&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Accessors
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
    // Implicit conversion to raw pointer for convenience
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }
    
    // Copy data from host to device
    void copyFrom(const T* host_data) {
        cudaError_t err = cudaMemcpy(
            ptr_, host_data, 
            size_ * sizeof(T), 
            cudaMemcpyHostToDevice
        );
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(err)
            );
        }
    }
    
    // Copy data from device to host
    void copyTo(T* host_data) const {
        cudaError_t err = cudaMemcpy(
            host_data, ptr_, 
            size_ * sizeof(T), 
            cudaMemcpyDeviceToHost
        );
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(err)
            );
        }
    }
    
    // Fill GPU memory with a value
    void fill(T value) {
        // Note: cudaMemset only works for single bytes
        // For general types, we'd need a kernel
        if (sizeof(T) == 1) {
            cudaMemset(ptr_, static_cast<int>(value), size_ * sizeof(T));
        } else {
            throw std::runtime_error("fill() not implemented for this type, use a kernel");
        }
    }
};
