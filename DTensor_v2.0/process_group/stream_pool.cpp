#include "stream_pool.h"
#include <stdexcept>
#include <thread>

namespace dtensor {

// =============================================================================
// StreamHandle Implementation
// =============================================================================

StreamHandle::StreamHandle(cudaStream_t stream, StreamPool* pool)
    : stream_(stream), pool_(pool), released_(false) {
}

StreamHandle::~StreamHandle() {
    if (!released_ && pool_ != nullptr) {
        pool_->release(stream_);
    }
}

StreamHandle::StreamHandle(StreamHandle&& other) noexcept
    : stream_(other.stream_), pool_(other.pool_), released_(other.released_) {
    other.released_ = true;  // Prevent double-release
}

StreamHandle& StreamHandle::operator=(StreamHandle&& other) noexcept {
    if (this != &other) {
        // Release current stream if not already released
        if (!released_ && pool_ != nullptr) {
            pool_->release(stream_);
        }
        
        stream_ = other.stream_;
        pool_ = other.pool_;
        released_ = other.released_;
        
        other.released_ = true;  // Prevent double-release
    }
    return *this;
}

// =============================================================================
// StreamPool Implementation
// =============================================================================

StreamPool::StreamPool(int device, size_t num_streams)
    : device_(device) {
    
    if (num_streams == 0) {
        throw std::invalid_argument("StreamPool requires at least 1 stream");
    }
    
    CUDA_CHECK(cudaSetDevice(device_));
    
    // Create all streams upfront
    streams_.reserve(num_streams);
    for (size_t i = 0; i < num_streams; ++i) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        streams_.push_back(stream);
        available_.push(stream);
    }
    
    std::cout << "[StreamPool] Created " << num_streams 
              << " streams on device " << device_ << std::endl;
}

StreamPool::~StreamPool() {
    CUDA_CHECK(cudaSetDevice(device_));
    
    // Destroy all streams
    for (cudaStream_t stream : streams_) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    
    std::cout << "[StreamPool] Destroyed " << streams_.size() 
              << " streams on device " << device_ << std::endl;
}

StreamHandle StreamPool::acquire(StreamPriority priority) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait for available stream (simple blocking for now)
    // In production, could add timeout or priority queue
    while (available_.empty()) {
        lock.unlock();
        // Yield CPU and retry - in production use condition variable
        std::this_thread::yield();
        lock.lock();
    }
    
    cudaStream_t stream = available_.front();
    available_.pop();
    
    return StreamHandle(stream, this);
}

cudaStream_t StreamPool::get_default_stream() const {
    // Return the first stream as the default
    return streams_[0];
}

void StreamPool::synchronize_all() {
    CUDA_CHECK(cudaSetDevice(device_));
    
    for (cudaStream_t stream : streams_) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

size_t StreamPool::available_streams() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_.size();
}

void StreamPool::release(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    available_.push(stream);
}

} // namespace dtensor
