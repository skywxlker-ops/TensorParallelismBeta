#include "process_group.h"
#include "error_handler.h"
#include <iostream>
#include <type_traits>

// Convenient macro aliases
#define CUDA_CHECK(call) CUDA_CHECK_THROW(call)
#define NCCL_CHECK(call) NCCL_CHECK_THROW(call)

// ---------------- Work ----------------
Work::Work(cudaStream_t stream)
    : stream_(stream), completed_(false), success_(true) {
    CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}

Work::~Work() {
    CUDA_CHECK(cudaEventDestroy(event_));
}

void Work::markCompleted(bool success) {
    success_ = success;
    completed_ = true;
    CUDA_CHECK(cudaEventRecord(event_, stream_));
}

bool Work::wait() {
    if (!completed_) return false;
    CUDA_CHECK(cudaEventSynchronize(event_));
    return success_;
}

// ---------------- ProcessGroup ----------------
ProcessGroup::ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id)
    : rank_(rank), world_size_(world_size), device_(device) {
    CUDA_CHECK(cudaSetDevice(device_));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    NCCL_WITH_RETRY(ncclCommInitRank(&comm_, world_size_, id, rank_), 3);
}

ProcessGroup::~ProcessGroup() {
    std::cerr << "[ProcessGroup] Destroyed" << std::endl;
    ncclCommDestroy(comm_);
    cudaStreamDestroy(stream_);
}

// ------------------------------------------------
// Template function definitions
// ------------------------------------------------
template <typename T>
std::shared_ptr<Work> ProcessGroup::allReduce(T* data, size_t count, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    NCCL_WITH_RETRY(ncclAllReduce(data, data, count, dtype, ncclSum, comm_, stream_), 3);
    work->markCompleted(true);
    return work;
}

template <typename T>
std::shared_ptr<Work> ProcessGroup::reduceScatter(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    NCCL_WITH_RETRY(ncclReduceScatter(send_buf, recv_buf, count_per_rank, dtype, ncclSum, comm_, stream_), 3);
    work->markCompleted(true);
    return work;
}

template <typename T>
std::shared_ptr<Work> ProcessGroup::allGather(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    NCCL_WITH_RETRY(ncclAllGather(send_buf, recv_buf, count_per_rank, dtype, comm_, stream_), 3);
    work->markCompleted(true);
    return work;
}

template <typename T>
std::shared_ptr<Work> ProcessGroup::broadcast(T* data, size_t count, int root, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    NCCL_WITH_RETRY(ncclBroadcast(data, data, count, dtype, root, comm_, stream_), 3);
    work->markCompleted(true);
    return work;
}

// ------------------------------------------------
// Point-to-Point: Send
// ------------------------------------------------
template <typename T>
std::shared_ptr<Work> ProcessGroup::send(T* data, size_t count, int dest, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(data, count, dtype, dest, comm_, stream_));
    NCCL_CHECK(ncclGroupEnd());
    work->markCompleted(true);
    return work;
}

// ------------------------------------------------
// Point-to-Point: Recv
// ------------------------------------------------
template <typename T>
std::shared_ptr<Work> ProcessGroup::recv(T* data, size_t count, int src, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclRecv(data, count, dtype, src, comm_, stream_));
    NCCL_CHECK(ncclGroupEnd());
    work->markCompleted(true);
    return work;
}

// ------------------------------------------------
// Scatter: Root sends different chunks to each rank
// ------------------------------------------------
template <typename T>
std::shared_ptr<Work> ProcessGroup::scatter(T* send_buf, T* recv_buf, size_t recv_count, int root, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(stream_);
    
    NCCL_CHECK(ncclGroupStart());
    
    if (rank_ == root) {
        // Root sends to each rank
        for (int dest = 0; dest < world_size_; ++dest) {
            T* chunk = send_buf + (dest * recv_count);
            if (dest == root) {
                // Self-copy for root
                CUDA_CHECK(cudaMemcpyAsync(recv_buf, chunk, 
                                          recv_count * sizeof(T),
                                          cudaMemcpyDeviceToDevice, stream_));
            } else {
                NCCL_CHECK(ncclSend(chunk, recv_count, dtype, dest, comm_, stream_));
            }
        }
    } else {
        // Non-root receives
        NCCL_CHECK(ncclRecv(recv_buf, recv_count, dtype, root, comm_, stream_));
    }
    
    NCCL_CHECK(ncclGroupEnd());
    work->markCompleted(true);
    return work;
}

// -------------------------------------------------------------
// ✅ Force Template Instantiations + Explicit Exports
// -------------------------------------------------------------
#define INSTANTIATE_AND_EXPORT(T, NCTYPE) \
    template std::shared_ptr<Work> ProcessGroup::allReduce<T>(T*, size_t, ncclDataType_t); \
    template std::shared_ptr<Work> ProcessGroup::reduceScatter<T>(T*, T*, size_t, ncclDataType_t); \
    template std::shared_ptr<Work> ProcessGroup::allGather<T>(T*, T*, size_t, ncclDataType_t); \
    template std::shared_ptr<Work> ProcessGroup::broadcast<T>(T*, size_t, int, ncclDataType_t); \
    template std::shared_ptr<Work> ProcessGroup::send<T>(T*, size_t, int, ncclDataType_t); \
    template std::shared_ptr<Work> ProcessGroup::recv<T>(T*, size_t, int, ncclDataType_t); \
    template std::shared_ptr<Work> ProcessGroup::scatter<T>(T*, T*, size_t, int, ncclDataType_t); \
    extern "C" __attribute__((visibility("default"))) void _force_link_##NCTYPE() { \
        volatile auto f1 = &ProcessGroup::allReduce<T>; \
        volatile auto f2 = &ProcessGroup::reduceScatter<T>; \
        volatile auto f3 = &ProcessGroup::allGather<T>; \
        volatile auto f4 = &ProcessGroup::broadcast<T>; \
        volatile auto f5 = &ProcessGroup::send<T>; \
        volatile auto f6 = &ProcessGroup::recv<T>; \
        volatile auto f7 = &ProcessGroup::scatter<T>; \
        (void)f1; (void)f2; (void)f3; (void)f4; (void)f5; (void)f6; (void)f7; \
    }

INSTANTIATE_AND_EXPORT(float, float)
INSTANTIATE_AND_EXPORT(double, double)
INSTANTIATE_AND_EXPORT(int, int32)
INSTANTIATE_AND_EXPORT(long long, int64)
