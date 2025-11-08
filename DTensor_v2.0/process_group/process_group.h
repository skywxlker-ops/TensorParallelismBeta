#pragma once
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <memory>
#include <iostream>

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
    std::shared_ptr<Work> allReduce(T* data, size_t count, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> reduceScatter(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> allGather(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> broadcast(T* data, size_t count, int root, ncclDataType_t dtype);

    cudaStream_t getStream() const { return stream_; }
    ncclComm_t getComm() const { return comm_; }

    // Accessors for PyBind11
    int getRank() const { return rank_; }
    int getWorldSize() const { return world_size_; }
    int getDevice() const { return device_; }

private:
    int rank_, world_size_, device_;
    ncclComm_t comm_;
    cudaStream_t stream_;
};

// === NCCL Type helper ===
inline ncclDataType_t getNcclType(const std::string& dtype) {
    if (dtype == "float32") return ncclFloat;
    if (dtype == "float64") return ncclDouble;
    if (dtype == "int32") return ncclInt;
    if (dtype == "int64") return ncclInt64;
    std::cerr << "Unsupported dtype for NCCL: " << dtype << std::endl;
    std::exit(1);
}
