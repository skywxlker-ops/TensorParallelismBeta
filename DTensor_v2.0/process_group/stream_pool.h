#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <mutex>
#include <memory>
#include <iostream>
#include <cstdlib>

// =============================================================================
// CUDA Error Checking (reuse from process_group.h)
// =============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "[CUDA Error] " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

// =============================================================================
// StreamPool: Thread-safe CUDA stream pool with RAII semantics
// =============================================================================

namespace dtensor {

/// Priority for stream allocation
enum class StreamPriority {
    HIGH,    // For critical operations (e.g., communication)
    NORMAL,  // For regular operations (e.g., computation)
    LOW      // For background operations (e.g., memory transfers)
};

// Forward declaration
class StreamPool;

/// RAII handle for automatic stream release
class StreamHandle {
public:
    StreamHandle(cudaStream_t stream, StreamPool* pool);
    ~StreamHandle();
    
    // No copy (move only)
    StreamHandle(const StreamHandle&) = delete;
    StreamHandle& operator=(const StreamHandle&) = delete;
    
    // Move semantics
    StreamHandle(StreamHandle&& other) noexcept;
    StreamHandle& operator=(StreamHandle&& other) noexcept;
    
    /// Get the underlying CUDA stream
    cudaStream_t get() const { return stream_; }
    
    /// Implicit conversion to cudaStream_t for convenience
    operator cudaStream_t() const { return stream_; }

private:
    cudaStream_t stream_;
    StreamPool* pool_;
    bool released_;
    
    friend class StreamPool;
};

/// Thread-safe pool of CUDA streams for efficient reuse
class StreamPool {
public:
    /**
     * Create a stream pool for the specified device
     * @param device CUDA device ID
     * @param num_streams Number of streams to maintain in the pool (default: 3)
     *                    Recommended: 3 streams (compute, communication, memory)
     */
    explicit StreamPool(int device, size_t num_streams = 3);
    
    /// Destructor - destroys all streams
    ~StreamPool();
    
    // No copy, no move (singleton-like behavior per ProcessGroup)
    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;
    
    /**
     * Acquire a stream from the pool (blocking if none available)
     * @param priority Priority hint for stream selection (currently unused, reserved)
     * @return RAII handle that auto-releases stream on destruction
     */
    StreamHandle acquire(StreamPriority priority = StreamPriority::NORMAL);
    
    /**
     * Get the default stream (stream 0, always available)
     * @return Default CUDA stream (non-blocking)
     */
    cudaStream_t get_default_stream() const;
    
    /**
     * Synchronize all streams in the pool
     * Blocks until all operations on all streams complete
     */
    void synchronize_all();
    
    /**
     * Get statistics
     */
    size_t total_streams() const { return streams_.size(); }
    size_t available_streams() const;
    
    /// Get device ID
    int device() const { return device_; }

private:
    // Internal method to release a stream back to the pool
    void release(cudaStream_t stream);
    
    int device_;
    std::vector<cudaStream_t> streams_;      // All streams owned by pool
    std::queue<cudaStream_t> available_;     // Available streams for reuse
    mutable std::mutex mutex_;               // Thread-safe access
    
    friend class StreamHandle;
};

} // namespace dtensor
