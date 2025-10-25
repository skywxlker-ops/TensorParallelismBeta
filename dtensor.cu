#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <iomanip>

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
        ncclCommInitRank(&comm_, world_size_, id, rank_);
    }

    ~ProcessGroup() {
        ncclCommDestroy(comm_);
        cudaStreamDestroy(all_reduce_stream_);
    }

    template<typename T>
    std::shared_ptr<Work> allReduce(T* data, size_t count, ncclDataType_t dtype) {
        auto work = std::make_shared<Work>(all_reduce_stream_);
        ncclAllReduce(data, data, count, dtype, ncclSum, comm_, all_reduce_stream_);
        work->markCompleted(true);
        return work;
    }

private:
    int rank_, world_size_, device_;
    ncclComm_t comm_;
    cudaStream_t all_reduce_stream_;
};

// ---------------- DTensor ----------------
class DTensor {
public:
    DTensor(int world_size, int slice_size)
        : world_size_(world_size), slice_size_(slice_size)
    {
        total_size_ = world_size_ * slice_size_;
        h_data_.resize(total_size_);
        for (int i = 0; i < world_size_; i++)
            for (int j = 0; j < slice_size_; j++)
                h_data_[i*slice_size_ + j] = float(i*10); // init rank*10

        cudaMalloc(&d_data_, total_size_ * sizeof(float));
        cudaMemcpy(d_data_, h_data_.data(), total_size_ * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~DTensor() { cudaFree(d_data_); }

    void print() {
        for (int i = 0; i < world_size_; i++) {
            std::cout << "[Slice " << i << "] ";
            for (int j = 0; j < slice_size_; j++)
                std::cout << h_data_[i*slice_size_ + j] << " ";
            std::cout << "\n";
        }
    }

    float* deviceData() { return d_data_; }
    size_t totalSize() const { return total_size_; }

    void copyDeviceToHost() {
        cudaMemcpy(h_data_.data(), d_data_, total_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    }

private:
    int world_size_, slice_size_;
    size_t total_size_;
    std::vector<float> h_data_;
    float* d_data_;
};

// ---------------- Worker ----------------
std::mutex g_io_mutex;

void worker(int rank, int world_size, const ncclUniqueId &id) {
    ProcessGroup pg(rank, world_size, rank, id);
    DTensor dt(world_size, 8); // 8 elements per slice

    {
        std::lock_guard<std::mutex> lg(g_io_mutex);
        std::cout << "==== Before AllReduce ====\n";
        dt.print();
    }

    // One big AllReduce over entire buffer
    auto w = pg.allReduce(dt.deviceData(), dt.totalSize(), ncclFloat32);
    w->wait();

    dt.copyDeviceToHost();
    {
        std::lock_guard<std::mutex> lg(g_io_mutex);
        std::cout << "==== After AllReduce ====\n";
        dt.print();
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
