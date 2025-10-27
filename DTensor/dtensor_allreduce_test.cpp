#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <cmath>

// ---------------- Work ----------------
class Work {
public:
    explicit Work(cudaStream_t stream)
        : stream_(stream), completed_(false), success_(true) {
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
    bool completed_, success_;
};

// ---------------- ProcessGroup ----------------
class ProcessGroup {
public:
    ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id)
        : rank_(rank), world_size_(world_size), device_(device) {
        cudaSetDevice(device_);
        cudaStreamCreate(&stream_);
        ncclCommInitRank(&comm_, world_size_, id, rank_);
    }

    ~ProcessGroup() {
        ncclCommDestroy(comm_);
        cudaStreamDestroy(stream_);
    }

    template<typename T>
    std::shared_ptr<Work> allReduce(T* data, size_t count, ncclDataType_t dtype) {
        auto work = std::make_shared<Work>(stream_);
        ncclAllReduce(data, data, count, dtype, ncclSum, comm_, stream_);
        work->markCompleted(true);
        return work;
    }

private:
    int rank_, world_size_, device_;
    ncclComm_t comm_;
    cudaStream_t stream_;
};

// ---------------- DTensor ----------------
class DTensor {
public:
    DTensor(int world_size, int slice_size, int rank)
        : world_size_(world_size), slice_size_(slice_size), rank_(rank) {
        h_data_.resize(slice_size_);
        for (int j = 0; j < slice_size_; ++j)
            h_data_[j] = float(rank_ * slice_size_ + j); // unique per rank

        cudaMalloc(&d_data_, slice_size_ * sizeof(float));
        cudaMemcpy(d_data_, h_data_.data(), slice_size_ * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    ~DTensor() {
        if (d_data_) cudaFree(d_data_);
    }

    float* deviceData() { return d_data_; }

    std::vector<float>& hostData() { return h_data_; }
    const std::vector<float>& hostData() const { return h_data_; }

    size_t size() const { return slice_size_; }

    void copyDeviceToHost() {
        cudaMemcpy(h_data_.data(), d_data_, slice_size_ * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "[Rank " << rank_ << "] ";
        for (float v : h_data_) ss << v << " ";
        return ss.str();
    }

private:
    int world_size_, slice_size_, rank_;
    std::vector<float> h_data_;
    float* d_data_ = nullptr;
};

// ---------------- Utility: Ordered Print ----------------
void printSectionHeader(int rank, const std::string &header) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << header << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void printAllRanks(int rank, int world_size, const std::string &data) {
    std::vector<int> sizes(world_size);
    int my_size = data.size();
    MPI_Gather(&my_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<std::string> all(world_size);
        all[0] = data;
        for (int r = 1; r < world_size; ++r) {
            all[r].resize(sizes[r]);
            MPI_Recv(&all[r][0], sizes[r], MPI_CHAR, r, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
        for (int r = 0; r < world_size; ++r)
            std::cout << all[r] << "\n";
        std::cout.flush();
    } else {
        MPI_Send(data.c_str(), my_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ---------------- Verify correctness ----------------
bool verifyAllReduce(const DTensor& tensor, int world_size) {
    const auto& host = tensor.hostData();
    std::vector<float> expected(host.size());
    for (int i = 0; i < tensor.size(); ++i) {
        float sum = 0;
        for (int r = 0; r < world_size; ++r)
            sum += float(r * tensor.size() + i);
        expected[i] = sum;
    }

    const float tol = 1e-5f;
    for (int i = 0; i < tensor.size(); ++i)
        if (fabs(host[i] - expected[i]) > tol)
            return false;
    return true;
}

// ---------------- Worker ----------------
void worker(int rank, int world_size, const ncclUniqueId &id, int slice_size) {
    ProcessGroup pg(rank, world_size, rank, id);
    DTensor tensor(world_size, slice_size, rank);

    printSectionHeader(rank, "==== BEFORE ALLREDUCE ====");
    printAllRanks(rank, world_size, tensor.toString());

    // AllReduce test
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    pg.allReduce(tensor.deviceData(), tensor.size(), ncclFloat32)->wait();
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    tensor.copyDeviceToHost();
    printSectionHeader(rank, "==== AFTER ALLREDUCE ====");
    printAllRanks(rank, world_size, tensor.toString());

    if (rank == 0) {
        std::cout << "\nAllReduce time: " << (end - start) * 1000 << " ms\n";
        std::cout << (verifyAllReduce(tensor, world_size)
                          ? "AllReduce verified successfully!\n"
                          : "AllReduce mismatch detected!\n");
    }
}

// ---------------- Main ----------------
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Starting AllReduce test with " << world_size << " processes\n";
    }

    int slice_size = (argc > 1) ? atoi(argv[1]) : 8;
    worker(rank, world_size, id, slice_size);

    MPI_Finalize();
    return 0;
}




