#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <iostream>

/**
 * StreamPool: Manages a pool of CUDA streams for concurrent execution
 * 
 * Purpose:
 * - Enable overlap of computation and communication
 * - Provide separate streams for different operation types
 * - Manage stream lifecycle and synchronization
 */
class StreamPool {
public:
    /**
     * Create a stream pool with specified number of streams
     * @param device CUDA device ID
     * @param num_streams Number of streams to create (default: 4)
     */
    explicit StreamPool(int device, int num_streams = 4)
        : device_(device), num_streams_(num_streams) {
        cudaSetDevice(device_);
        
        streams_.resize(num_streams_);
        events_.resize(num_streams_);
        
        for (int i = 0; i < num_streams_; ++i) {
            cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming);
        }
        
        // Pre-defined stream indices
        COMPUTE_STREAM_ID = 0;
        COMM_STREAM_ID = 1;
        DATA_STREAM_ID = 2;
        GENERAL_STREAM_ID = 3;
    }
    
    ~StreamPool() {
        cudaSetDevice(device_);
        for (int i = 0; i < num_streams_; ++i) {
            cudaStreamSynchronize(streams_[i]);
            cudaStreamDestroy(streams_[i]);
            cudaEventDestroy(events_[i]);
        }
    }
    
    // Get specific streams for different purposes
    cudaStream_t getComputeStream() const { return streams_[COMPUTE_STREAM_ID]; }
    cudaStream_t getCommStream() const { return streams_[COMM_STREAM_ID]; }
    cudaStream_t getDataStream() const { return streams_[DATA_STREAM_ID]; }
    cudaStream_t getGeneralStream() const { return streams_[GENERAL_STREAM_ID]; }
    
    // Get stream by ID
    cudaStream_t getStream(int id) const {
        if (id < 0 || id >= num_streams_) {
            std::cerr << "[StreamPool] Invalid stream ID: " << id << std::endl;
            return streams_[GENERAL_STREAM_ID];
        }
        return streams_[id];
    }
    
    // Get event by ID for synchronization
    cudaEvent_t getEvent(int id) const {
        if (id < 0 || id >= num_streams_) {
            std::cerr << "[StreamPool] Invalid event ID: " << id << std::endl;
            return events_[GENERAL_STREAM_ID];
        }
        return events_[id];
    }
    
    int getNumStreams() const { return num_streams_; }
    
    /**
     * Record event on stream1, then make stream2 wait for it
     * This creates a dependency: stream2 waits until stream1 reaches this point
     */
    void synchronizeStreams(int stream1_id, int stream2_id) {
        cudaEventRecord(events_[stream1_id], streams_[stream1_id]);
        cudaStreamWaitEvent(streams_[stream2_id], events_[stream1_id], 0);
    }
    
    /**
     * Synchronize all streams
     */
    void synchronizeAll() {
        for (int i = 0; i < num_streams_; ++i) {
            cudaStreamSynchronize(streams_[i]);
        }
    }
    
    // Stream IDs for semantic access
    int COMPUTE_STREAM_ID;  // For matmul, activations
    int COMM_STREAM_ID;     // For NCCL collectives
    int DATA_STREAM_ID;     // For data transfers
    int GENERAL_STREAM_ID;  // For general operations
    
private:
    int device_;
    int num_streams_;
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
};
