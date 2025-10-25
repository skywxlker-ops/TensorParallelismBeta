// // dtensor_allreduce.cu
// #include <cuda_runtime.h>
// #include <nccl.h>
// #include <iostream>
// #include <vector>
// #include <thread>
// #include <memory>
// #include <mutex>
// #include <chrono>
// #include <iomanip>

// // ---------------- Work ----------------
// class Work {
// public:
//     explicit Work(cudaStream_t stream) : stream_(stream), completed_(false), success_(true) {
//         cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
//     }
//     ~Work() { cudaEventDestroy(event_); }

//     void markCompleted(bool success = true) {
//         success_ = success;
//         completed_ = true;
//         cudaEventRecord(event_, stream_);
//     }

//     bool wait() {
//         if (!completed_) return false;
//         cudaEventSynchronize(event_);
//         return success_;
//     }

// private:
//     cudaStream_t stream_;
//     cudaEvent_t event_;
//     bool completed_;
//     bool success_;
// };

// // ---------------- ProcessGroup ----------------
// class ProcessGroup {
// public:
//     ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id)
//         : rank_(rank), world_size_(world_size), device_(device) 
//     {
//         cudaSetDevice(device_);
//         cudaStreamCreate(&all_reduce_stream_);
//         ncclCommInitRank(&comm_, world_size_, id, rank_);
//     }

//     ~ProcessGroup() {
//         ncclCommDestroy(comm_);
//         cudaStreamDestroy(all_reduce_stream_);
//     }

//     template<typename T>
//     std::shared_ptr<Work> allReduce(T* data, size_t count, ncclDataType_t dtype) {
//         auto work = std::make_shared<Work>(all_reduce_stream_);
//         ncclAllReduce(data, data, count, dtype, ncclSum, comm_, all_reduce_stream_);
//         work->markCompleted(true);
//         return work;
//     }

//     int rank() const { return rank_; }
//     int world_size() const { return world_size_; }

// private:
//     int rank_, world_size_, device_;
//     ncclComm_t comm_;
//     cudaStream_t all_reduce_stream_;
// };

// // ---------------- DTensor ----------------
// class DTensor {
// public:
//     DTensor(int world_size, int slice_size) 
//         : world_size_(world_size), slice_size_(slice_size)
//     {
//         for (int i = 0; i < world_size_; i++) {
//             std::vector<float> slice(slice_size_, float(i*10)); // initialize with rank*10
//             slices_.push_back(slice);
//         }

//         // Allocate device memory
//         d_slices_.resize(world_size_);
//         for (int i = 0; i < world_size_; i++) {
//             cudaMalloc(&d_slices_[i], slice_size_ * sizeof(float));
//             cudaMemcpy(d_slices_[i], slices_[i].data(), slice_size_ * sizeof(float), cudaMemcpyHostToDevice);
//         }
//     }

//     ~DTensor() {
//         for (auto ptr : d_slices_) cudaFree(ptr);
//     }

//     void printSlices() {
//         for (int i = 0; i < world_size_; i++) {
//             std::cout << "[Rank " << i << "] ";
//             for (float v : slices_[i]) std::cout << v << " ";
//             std::cout << "\n";
//         }
//     }

//     int getWorldSize() const { return world_size_; }
//     int getSliceSize() const { return slice_size_; }

//     float* deviceSlice(int rank) { return d_slices_[rank]; }

//     void copyDeviceToHost() {
//         for (int i = 0; i < world_size_; i++)
//             cudaMemcpy(slices_[i].data(), d_slices_[i], slice_size_ * sizeof(float), cudaMemcpyDeviceToHost);
//     }

// private:
//     int world_size_, slice_size_;
//     std::vector<std::vector<float>> slices_;
//     std::vector<float*> d_slices_;
// };

// // ---------------- Worker ----------------
// std::mutex g_io_mutex;

// void worker(int rank, int world_size, const ncclUniqueId &id) {
//     ProcessGroup pg(rank, world_size, rank, id);
//     DTensor dt(world_size, 4);

//     {
//         std::lock_guard<std::mutex> lg(g_io_mutex);
//         std::cout << "==== Before NCCL AllReduce ====\n";
//         dt.printSlices();
//     }

//     // AllReduce each slice across ranks
//     for (int i = 0; i < world_size; i++) {
//         auto w = pg.allReduce(dt.deviceSlice(i), dt.getSliceSize(), ncclFloat32);
//         w->wait();
//     }

//     dt.copyDeviceToHost();

//     {
//         std::lock_guard<std::mutex> lg(g_io_mutex);
//         std::cout << "==== After NCCL AllReduce ====\n";
//         dt.printSlices();
//     }
// }

// // ---------------- Main ----------------
// int main() {
//     int world_size = 2;
//     ncclUniqueId id;
//     ncclGetUniqueId(&id);

//     std::vector<std::thread> threads;
//     for (int r = 0; r < world_size; ++r)
//         threads.emplace_back(worker, r, world_size, std::ref(id));

//     for (auto &t : threads) t.join();
//     return 0;
// }


// dtensor_full.cu
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

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }

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
        // Host slices
        for (int i = 0; i < world_size_; i++) {
            std::vector<float> slice(slice_size_, float(i*10)); // rank*10
            slices_.push_back(slice);
        }

        // Device slices
        d_slices_.resize(world_size_);
        for (int i = 0; i < world_size_; i++) {
            cudaMalloc(&d_slices_[i], slice_size_ * sizeof(float));
            cudaMemcpy(d_slices_[i], slices_[i].data(), slice_size_ * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    ~DTensor() {
        for (auto ptr : d_slices_) cudaFree(ptr);
    }

    void printSlices() {
        for (int i = 0; i < world_size_; i++) {
            std::cout << "[Rank " << i << "] ";
            for (float v : slices_[i]) std::cout << v << " ";
            std::cout << "\n";
        }
    }

    int getWorldSize() const { return world_size_; }
    int getSliceSize() const { return slice_size_; }

    float* deviceSlice(int rank) { return d_slices_[rank]; }

    void copyDeviceToHost() {
        for (int i = 0; i < world_size_; i++)
            cudaMemcpy(slices_[i].data(), d_slices_[i], slice_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    }

private:
    int world_size_, slice_size_;
    std::vector<std::vector<float>> slices_;
    std::vector<float*> d_slices_;
};

// ---------------- Worker ----------------
std::mutex g_io_mutex;

void worker(int rank, int world_size, const ncclUniqueId &id) {
    ProcessGroup pg(rank, world_size, rank, id);
    DTensor dt(world_size, 4);

    {
        std::lock_guard<std::mutex> lg(g_io_mutex);
        std::cout << "==== Before NCCL AllReduce ====\n";
        dt.printSlices();
    }

    // AllReduce each slice across ranks
    for (int i = 0; i < world_size; i++) {
        auto w = pg.allReduce(dt.deviceSlice(i), dt.getSliceSize(), ncclFloat32);
        w->wait();
    }

    dt.copyDeviceToHost();

    {
        std::lock_guard<std::mutex> lg(g_io_mutex);
        std::cout << "==== After NCCL AllReduce ====\n";
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





