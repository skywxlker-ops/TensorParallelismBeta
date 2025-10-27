// DTensor/dtensor_core.cpp
#include "dtensor_core.h"
#include <iostream>
#include <vector>
#include <memory>

// Work implementation
Work::Work(cudaStream_t stream) : stream_(stream), completed_(false), success_(true) {
    cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
}

Work::~Work() { 
    cudaEventDestroy(event_); 
}

void Work::markCompleted(bool success) {
    success_ = success;
    completed_ = true;
    cudaEventRecord(event_, stream_);
}

bool Work::wait() {
    if (!completed_) return false;
    cudaEventSynchronize(event_);
    return success_;
}

// ProcessGroup implementation
ProcessGroup::ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id)
    : rank_(rank), world_size_(world_size), device_(device) 
{
    cudaSetDevice(device_);
    cudaStreamCreate(&all_reduce_stream_);
    cudaStreamCreate(&reduce_scatter_stream_);
    cudaStreamCreate(&all_gather_stream_);
    cudaStreamCreate(&broadcast_stream_);

    ncclCommInitRank(&comm_, world_size_, id, rank_);
}

ProcessGroup::~ProcessGroup() {
    ncclCommDestroy(comm_);
    cudaStreamDestroy(all_reduce_stream_);
    cudaStreamDestroy(reduce_scatter_stream_);
    cudaStreamDestroy(all_gather_stream_);
    cudaStreamDestroy(broadcast_stream_);
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::all_reduce(T* data, size_t count, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(all_reduce_stream_);
    ncclAllReduce(data, data, count, dtype, ncclSum, comm_, all_reduce_stream_);
    work->markCompleted(true);
    return work;
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::reduce_scatter(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(reduce_scatter_stream_);
    ncclReduceScatter(send_buf, recv_buf, count_per_rank, dtype, ncclSum, comm_, reduce_scatter_stream_);
    work->markCompleted(true);
    return work;
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::all_gather(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(all_gather_stream_);
    ncclAllGather(send_buf, recv_buf, count_per_rank, dtype, comm_, all_gather_stream_);
    work->markCompleted(true);
    return work;
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::broadcast(T* data, size_t count, int root, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(broadcast_stream_);
    ncclBroadcast(data, data, count, dtype, root, comm_, broadcast_stream_);
    work->markCompleted(true);
    return work;
}

// Explicit template instantiations
template std::shared_ptr<Work> ProcessGroup::all_reduce<float>(float*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::reduce_scatter<float>(float*, float*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::all_gather<float>(float*, float*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::broadcast<float>(float*, size_t, int, ncclDataType_t);

// DTensor implementation
DTensor::DTensor(int world_size, size_t slice_size) 
    : world_size_(world_size), slice_size_(slice_size) {
    slices_.resize(world_size_);
    d_slices_.resize(world_size_);

    for (int i = 0; i < world_size_; ++i) {
        std::vector<float> slice(slice_size_);
        for (size_t j = 0; j < slice_size_; ++j)
            slice[j] = float(i * slice_size_ + j); // unique per rank
        slices_[i] = slice;

        cudaMalloc(&d_slices_[i], slice_size_ * sizeof(float));
        cudaMemcpy(d_slices_[i], slices_[i].data(), slice_size_ * sizeof(float), cudaMemcpyHostToDevice);
    }
}

DTensor::~DTensor() {
    for (auto ptr : d_slices_) {
        if (ptr) cudaFree(ptr);
    }
}

float* DTensor::deviceSlice(int rank) { 
    return d_slices_[rank]; 
}

size_t DTensor::sliceSize() const { 
    return slice_size_; 
}

void DTensor::copyDeviceToHost() {
    for (int i = 0; i < world_size_; ++i) {
        cudaMemcpy(slices_[i].data(), d_slices_[i], slice_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

void DTensor::printSlices() {
    for (int i = 0; i < world_size_; ++i) {
        std::cout << "[Slice " << i << "] ";
        for (float v : slices_[i]) std::cout << v << " ";
        std::cout << "\n";
    }
}