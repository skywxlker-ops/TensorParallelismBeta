// DTensor/dtensor_core.h
#ifndef DTENSOR_CORE_H
#define DTENSOR_CORE_H

#include <cuda_runtime.h>
#include <nccl.h>
#include <memory>
#include <vector>

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

class ProcessGroup {
public:
    ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id);
    ~ProcessGroup();

    template<typename T>
    std::shared_ptr<Work> all_reduce(T* data, size_t count, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> reduce_scatter(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> all_gather(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> broadcast(T* data, size_t count, int root, ncclDataType_t dtype);

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }

private:
    int rank_, world_size_, device_;
    ncclComm_t comm_;
    cudaStream_t all_reduce_stream_;
    cudaStream_t reduce_scatter_stream_;
    cudaStream_t all_gather_stream_;
    cudaStream_t broadcast_stream_;
};

class DTensor {
public:
    DTensor(int world_size, size_t slice_size);
    ~DTensor();

    float* deviceSlice(int rank);
    size_t sliceSize() const;
    void copyDeviceToHost();
    void printSlices();

private:
    int world_size_;
    size_t slice_size_;
    std::vector<std::vector<float>> slices_;
    std::vector<float*> d_slices_;
};

#endif // DTENSOR_CORE_H