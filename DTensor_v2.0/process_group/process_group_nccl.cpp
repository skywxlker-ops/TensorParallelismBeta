#include "process_group.h"
#include <iostream>
#include <type_traits>

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
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    ncclCommInitRank(&comm_, world_size_, id, rank_);
}

ProcessGroup::~ProcessGroup() {
    std::cerr << "[ProcessGroup] Destroyed" << std::endl;
    ncclCommDestroy(comm_);
    cudaStreamDestroy(stream_);
}



template <typename T>
std::shared_ptr<Work> ProcessGroup::allReduce(T* data, size_t count, ncclDataType_t dtype, ncclRedOp_t op) {
    auto work = std::make_shared<Work>(stream_);
    ncclAllReduce(data, data, count, dtype, op, comm_, stream_);
    work->markCompleted(true);
    return work;
}


template <typename T>
std::shared_ptr<Work> ProcessGroup::reduceScatter(T* data, size_t count_per_shard, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    ncclReduceScatter(data, data + rank_ * count_per_shard, count_per_shard, dtype, ncclSum, comm_, stream_);
    work->markCompleted(true);
    return work;
}

template <typename T>
std::shared_ptr<Work> ProcessGroup::allGather(T* data, size_t count_per_rank, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    ncclAllGather(data, data + rank_ * count_per_rank , count_per_rank, dtype, comm_, stream_);
    work->markCompleted(true);
    return work;
}

template <typename T>
std::shared_ptr<Work> ProcessGroup::broadcast(T* data, size_t count, int root, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    ncclBroadcast(data, data, count, dtype, root, comm_, stream_);
    work->markCompleted(true);
    return work;
}


template <typename T>
std::shared_ptr<Work> ProcessGroup::scatter(T* data, size_t count_per_shard, int root, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    ncclScatter(data, data + rank_ * count_per_shard, count_per_shard, dtype, root, comm_, stream_);
    work->markCompleted(true);
    return work;
}


#define INSTANTIATE_AND_EXPORT(T, NCTYPE) \
    template std::shared_ptr<Work> ProcessGroup::allReduce<T>(T*, size_t, ncclDataType_t, ncclRedOp_t); \
    template std::shared_ptr<Work> ProcessGroup::reduceScatter<T>(T*, size_t, ncclDataType_t); \
    template std::shared_ptr<Work> ProcessGroup::allGather<T>(T*, size_t, ncclDataType_t); \
    template std::shared_ptr<Work> ProcessGroup::broadcast<T>(T*, size_t, int, ncclDataType_t); \
    template std::shared_ptr<Work> ProcessGroup::scatter<T>(T*, size_t, int, ncclDataType_t); \
    extern "C" __attribute__((visibility("default"))) void _force_link_##NCTYPE() { \
        volatile auto f1 = &ProcessGroup::allReduce<T>; \
        volatile auto f2 = &ProcessGroup::reduceScatter<T>; \
        volatile auto f3 = &ProcessGroup::allGather<T>; \
        volatile auto f4 = &ProcessGroup::broadcast<T>; \
        volatile auto f5 = &ProcessGroup::scatter<T>; \
        (void)f1; (void)f2; (void)f3; (void)f4; (void)f5;\
    }

INSTANTIATE_AND_EXPORT(float, float)
INSTANTIATE_AND_EXPORT(double, double)
INSTANTIATE_AND_EXPORT(int, int32)
INSTANTIATE_AND_EXPORT(long long, int64)






