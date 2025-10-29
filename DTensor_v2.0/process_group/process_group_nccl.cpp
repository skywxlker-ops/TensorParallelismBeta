#include "process_group.h"
#include <iostream>

// ---------------- Work ----------------
Work::Work(cudaStream_t stream) 
    : stream_(stream), completed_(false), success_(true) {
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

// ---------------- ProcessGroup ----------------
ProcessGroup::ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id)
    : rank_(rank), world_size_(world_size), device_(device) {
    cudaSetDevice(device_);
    cudaStreamCreate(&stream_);
    ncclCommInitRank(&comm_, world_size_, id, rank_);
}

ProcessGroup::~ProcessGroup() {
    ncclCommDestroy(comm_);
    cudaStreamDestroy(stream_);
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::allReduce(T* data, size_t count, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    ncclAllReduce(data, data, count, dtype, ncclSum, comm_, stream_);
    work->markCompleted(true);
    return work;
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::reduceScatter(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    ncclReduceScatter(send_buf, recv_buf, count_per_rank, dtype, ncclSum, comm_, stream_);
    work->markCompleted(true);
    return work;
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::allGather(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    ncclAllGather(send_buf, recv_buf, count_per_rank, dtype, comm_, stream_);
    work->markCompleted(true);
    return work;
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::broadcast(T* data, size_t count, int root, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    ncclBroadcast(data, data, count, dtype, root, comm_, stream_);
    work->markCompleted(true);
    return work;
}

// Explicit instantiations for float (add more types if needed)
template std::shared_ptr<Work> ProcessGroup::allReduce<float>(float*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::reduceScatter<float>(float*, float*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::allGather<float>(float*, float*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::broadcast<float>(float*, size_t, int, ncclDataType_t);
