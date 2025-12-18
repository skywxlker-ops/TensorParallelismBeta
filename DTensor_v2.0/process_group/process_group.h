#pragma once
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <memory>
#include <iostream>
#include "stream_pool.h"

// ---------------- Work ----------------
class Work {
public:
    explicit Work(cudaStream_t stream);
    ~Work();

    void markCompleted(bool success = true);
    bool wait();

private:
    cudaStream_t stream_;
    cudaEvent_t event_;
    bool completed_;
    bool success_;
};


// ---------------- ProcessGroup ----------------
class ProcessGroup {
public:
    ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id);
    ~ProcessGroup();

    template<typename T>
    std::shared_ptr<Work> allReduce(const T* input, T* output, size_t count, ncclDataType_t dtype, ncclRedOp_t op = ncclSum);

    template<typename T>
    std::shared_ptr<Work> reduceScatter(const T* input, T* output, size_t count_per_shard, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> allGather(const T* input, T* output, size_t count_per_rank, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> broadcast(const T* input, T* output, size_t count, int root, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> scatter(const T* input, T* output, size_t count_per_shard, int root, ncclDataType_t dtype);

    // Stream access methods
    cudaStream_t getStream() const { return stream_; }  // Backward compatibility (returns comm stream)
    cudaStream_t getComputeStream() const { return stream_pool_->getComputeStream(); }
    cudaStream_t getCommStream() const { return stream_pool_->getCommStream(); }
    cudaStream_t getDataStream() const { return stream_pool_->getDataStream(); }
    
    std::shared_ptr<StreamPool> getStreamPool() const { return stream_pool_; }
    ncclComm_t getComm() const { return comm_; }

    // NCCL grouped operations support
    void startGroup();  // Begin NCCL group (ncclGroupStart)
    void endGroup();    // End NCCL group (ncclGroupEnd)
    bool inGroup() const { return in_group_; }
    
    int getRank() const { return rank_; }
    int getWorldSize() const { return world_size_; }
    int getDevice() const { return device_; }

private:
    int rank_, world_size_, device_;
    ncclComm_t comm_;
    cudaStream_t stream_;  // Keep for backward compatibility (points to comm stream)
    std::shared_ptr<StreamPool> stream_pool_;  // New: Stream pool for concurrent execution
    bool in_group_ = false;  // Track if currently in NCCL group
};

// NCCL Type helper 
inline ncclDataType_t getNcclType(const std::string& dtype) {
    if (dtype == "float32") return ncclFloat;
    if (dtype == "float64") return ncclDouble;
    if (dtype == "int32") return ncclInt;
    if (dtype == "int64") return ncclInt64;
    std::cerr << "Unsupported dtype for NCCL: " << dtype << std::endl;
    std::exit(1);
}
