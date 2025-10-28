#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// ---------------- Work ----------------
class Work {
public:
    explicit Work(cudaStream_t stream);
    ~Work();

    void markCompleted(bool success = true);
    bool wait();

private:
    cudaStream_t stream_;
    cudaEvent_t  event_{};
    bool         completed_{false};
    bool         success_{true};
};

// ---------------- ProcessGroup ----------------
class ProcessGroup {
public:
    ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id);
    ~ProcessGroup();

    // templated collectives must live in the header
    template<typename T>
    std::shared_ptr<Work> allReduce(T* data, size_t count, ncclDataType_t dtype) {
        auto work = std::make_shared<Work>(stream_);
        ncclAllReduce(data, data, count, dtype, ncclSum, comm_, stream_);
        work->markCompleted(true);
        return work;
    }

    template<typename T>
    std::shared_ptr<Work> reduceScatter(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
        auto work = std::make_shared<Work>(stream_);
        ncclReduceScatter(send_buf, recv_buf, count_per_rank, dtype, ncclSum, comm_, stream_);
        work->markCompleted(true);
        return work;
    }

    template<typename T>
    std::shared_ptr<Work> allGather(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
        auto work = std::make_shared<Work>(stream_);
        ncclAllGather(send_buf, recv_buf, count_per_rank, dtype, comm_, stream_);
        work->markCompleted(true);
        return work;
    }

    template<typename T>
    std::shared_ptr<Work> broadcast(T* data, size_t count, int root, ncclDataType_t dtype) {
        auto work = std::make_shared<Work>(stream_);
        ncclBroadcast(data, data, count, dtype, root, comm_, stream_);
        work->markCompleted(true);
        return work;
    }

    cudaStream_t getStream() { return stream_; }
    int rank() const { return rank_; }
    int worldSize() const { return world_size_; }

private:
    int          rank_{0};
    int          world_size_{1};
    int          device_{0};
    cudaStream_t stream_{};
    ncclComm_t   comm_{};
};

// ---------------- DTensor ----------------
class DTensor {
public:
    DTensor(int world_size, int slice_size, int rank);
    ~DTensor();

    float* deviceData();
    std::vector<float>& hostData();
    size_t size() const;

    void copyDeviceToHost();

    // (OPTIONAL) you can add copyFromHost/copyToHost helpers here too.

private:
    int world_size_{1};
    int slice_size_{0};
    int rank_{0};
    std::vector<float> h_data_;
    float* d_data_{nullptr};
};

// ---------------- Demo declarations (will be excluded from pybind build) -------------
#ifndef DTENSOR_PYBIND_BUILD
void worker(int rank, int world_size, const ncclUniqueId &id);
int main(int argc, char* argv[]);
#endif
