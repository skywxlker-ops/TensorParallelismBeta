// dtensor_full_collectives.cpp
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>

// ---------------- Work ----------------
class Work {
public:
    explicit Work(cudaStream_t stream) : stream_(stream), completed_(false), success_(true) {
        cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    }
    ~Work() { cudaEventDestroy(event_); }

    void markCompleted(bool success = true) {
        success_ = success;
        completed_ = true;
        cudaEventRecord(event_, stream_);
    }

    bool wait() {
        if (!completed_) return false;
        cudaEventSynchronize(event_);
        return success_;
    }

private:
    cudaStream_t stream_;
    cudaEvent_t event_;
    bool completed_;
    bool success_;
};

// ---------------- ProcessGroup ----------------
class ProcessGroup {
public:
    ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id)
        : rank_(rank), world_size_(world_size), device_(device) 
    {
        cudaSetDevice(device_);
        cudaStreamCreate(&all_reduce_stream_);
        cudaStreamCreate(&reduce_scatter_stream_);
        cudaStreamCreate(&all_gather_stream_);
        cudaStreamCreate(&broadcast_stream_);

        ncclCommInitRank(&comm_, world_size_, id, rank_);
    }

    ~ProcessGroup() {
        ncclCommDestroy(comm_);
        cudaStreamDestroy(all_reduce_stream_);
        cudaStreamDestroy(reduce_scatter_stream_);
        cudaStreamDestroy(all_gather_stream_);
        cudaStreamDestroy(broadcast_stream_);
    }

    template<typename T>
    std::shared_ptr<Work> all_reduce(T* data, size_t count, ncclDataType_t dtype) {
        auto work = std::make_shared<Work>(all_reduce_stream_);
        ncclAllReduce(data, data, count, dtype, ncclSum, comm_, all_reduce_stream_);
        work->markCompleted(true);
        return work;
    }

    template<typename T>
    std::shared_ptr<Work> reduce_scatter(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
        auto work = std::make_shared<Work>(reduce_scatter_stream_);
        ncclReduceScatter(send_buf, recv_buf, count_per_rank, dtype, ncclSum, comm_, reduce_scatter_stream_);
        work->markCompleted(true);
        return work;
    }

    template<typename T>
    std::shared_ptr<Work> all_gather(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
        auto work = std::make_shared<Work>(all_gather_stream_);
        ncclAllGather(send_buf, recv_buf, count_per_rank, dtype, comm_, all_gather_stream_);
        work->markCompleted(true);
        return work;
    }

    template<typename T>
    std::shared_ptr<Work> broadcast(T* data, size_t count, int root, ncclDataType_t dtype) {
        auto work = std::make_shared<Work>(broadcast_stream_);
        ncclBroadcast(data, data, count, dtype, root, comm_, broadcast_stream_);
        work->markCompleted(true);
        return work;
    }

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

// ---------------- DTensor ----------------
class DTensor {
public:
    DTensor(int world_size, int slice_size) : world_size_(world_size), slice_size_(slice_size) {
        for (int i = 0; i < world_size_; ++i) {
            std::vector<float> slice(slice_size_, float(i*10));
            slices_.push_back(slice);
        }

        d_slices_.resize(world_size_);
        for (int i = 0; i < world_size_; ++i) {
            cudaMalloc(&d_slices_[i], slice_size_ * sizeof(float));
            cudaMemcpy(d_slices_[i], slices_[i].data(), slice_size_ * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    ~DTensor() {
        for (auto ptr : d_slices_) cudaFree(ptr);
    }

    float* deviceSlice(int rank) { return d_slices_[rank]; }
    size_t sliceSize() const { return slice_size_; }

    void copyDeviceToHost() {
        for (int i = 0; i < world_size_; ++i)
            cudaMemcpy(slices_[i].data(), d_slices_[i], slice_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void printSlices() {
        for (int i = 0; i < world_size_; ++i) {
            std::cout << "[Slice " << i << "] ";
            for (float v : slices_[i]) std::cout << v << " ";
            std::cout << "\n";
        }
    }

private:
    int world_size_;
    size_t slice_size_;
    std::vector<std::vector<float>> slices_;
    std::vector<float*> d_slices_;
};

// ---------------- Worker ----------------
std::mutex g_io_mutex;

void worker(int rank, int world_size, const ncclUniqueId &id) {
    ProcessGroup pg(rank, world_size, rank, id);
    DTensor dt(world_size, 8);

    {
        std::lock_guard<std::mutex> lg(g_io_mutex);
        std::cout << "==== Before Any Collective ====\n";
        dt.printSlices();
    }

    // --- AllReduce ---
    for (int i = 0; i < world_size; ++i)
        pg.all_reduce(dt.deviceSlice(i), dt.sliceSize(), ncclFloat32)->wait();

    dt.copyDeviceToHost();
    {
        std::lock_guard<std::mutex> lg(g_io_mutex);
        std::cout << "==== After AllReduce ====\n";
        dt.printSlices();
    }

    // --- ReduceScatter ---
    for (int i = 0; i < world_size; ++i)
        pg.reduce_scatter(dt.deviceSlice(i), dt.deviceSlice(i), dt.sliceSize() / world_size, ncclFloat32)->wait();

    dt.copyDeviceToHost();
    {
        std::lock_guard<std::mutex> lg(g_io_mutex);
        std::cout << "==== After ReduceScatter ====\n";
        dt.printSlices();
    }

    // --- AllGather ---
    for (int i = 0; i < world_size; ++i)
        pg.all_gather(dt.deviceSlice(i), dt.deviceSlice(i), dt.sliceSize() / world_size, ncclFloat32)->wait();

    dt.copyDeviceToHost();
    {
        std::lock_guard<std::mutex> lg(g_io_mutex);
        std::cout << "==== After AllGather ====\n";
        dt.printSlices();
    }

    // --- Broadcast (root=0) ---
    for (int i = 0; i < world_size; ++i)
        pg.broadcast(dt.deviceSlice(i), dt.sliceSize(), 0, ncclFloat32)->wait();

    dt.copyDeviceToHost();
    {
        std::lock_guard<std::mutex> lg(g_io_mutex);
        std::cout << "==== After Broadcast ====\n";
        dt.printSlices();
    }
}

// ---------------- Main ----------------
int main() {
    int world_size = 2;
    ncclUniqueId id;
    ncclGetUniqueId(&id);

    std::vector<std::thread> threads;
    for (int r = 0; r < world_size; ++r)
        threads.emplace_back(worker, r, world_size, std::ref(id));

    for (auto &t : threads) t.join();
    return 0;
}
