#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>

// ---------------- Work ----------------
class Work {
public:
    explicit Work(cudaStream_t stream) : stream_(stream), completed_(false), success_(true) {
        cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    }
    ~Work() { 
        cudaEventDestroy(event_); 
    }

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

private:
    int rank_, world_size_, device_;
    ncclComm_t comm_;
    cudaStream_t stream_;
};

// ---------------- DTensor ----------------
class DTensor {
public:
    DTensor(int world_size, int slice_size, int rank) 
        : world_size_(world_size), slice_size_(slice_size), rank_(rank) 
    {
        // Each rank only manages its own slice
        h_data_.resize(slice_size_);
        for (int j = 0; j < slice_size_; ++j) {
            h_data_[j] = float(rank_ * slice_size_ + j); // unique per rank
        }
        
        cudaMalloc(&d_data_, slice_size_ * sizeof(float));
        cudaMemcpy(d_data_, h_data_.data(), slice_size_ * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~DTensor() {
        if (d_data_) cudaFree(d_data_);
    }

    float* deviceData() { return d_data_; }
    std::vector<float>& hostData() { return h_data_; }
    size_t size() const { return slice_size_; }

    void copyDeviceToHost() {
        cudaMemcpy(h_data_.data(), d_data_, slice_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    }

    std::string toString() {
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

// ---------------- Perfect Output Synchronization ----------------
void printSectionHeader(int rank, const std::string& header) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << header << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void printAllRanks(int rank, int world_size, const std::string& data) {
    // Collect all outputs at rank 0 and print in order
    std::vector<std::string> all_outputs(world_size);
    
    // First, gather the string lengths
    int my_size = data.size();
    std::vector<int> sizes(world_size);
    MPI_Gather(&my_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Then gather the strings
    if (rank == 0) {
        // Rank 0 receives all data
        all_outputs[0] = data;
        for (int r = 1; r < world_size; r++) {
            all_outputs[r].resize(sizes[r]);
            MPI_Recv(&all_outputs[r][0], sizes[r], MPI_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Print in rank order
        for (int r = 0; r < world_size; r++) {
            std::cout << all_outputs[r] << "\n";
        }
        std::cout.flush();
    } else {
        // Other ranks send to rank 0
        MPI_Send(data.c_str(), my_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// ---------------- Worker ----------------
void worker(int rank, int world_size, const ncclUniqueId &id) {
    int device = rank;
    ProcessGroup pg(rank, world_size, device, id);
    
    // Each rank has its own tensor slice
    DTensor tensor(world_size, 8, rank);

    // ==== Before Any Collective ====
    printSectionHeader(rank, "==== BEFORE ANY COLLECTIVE ====");
    printAllRanks(rank, world_size, tensor.toString());

    // --- AllReduce ---
    pg.allReduce(tensor.deviceData(), tensor.size(), ncclFloat32)->wait();
    tensor.copyDeviceToHost();
    
    printSectionHeader(rank, "==== AFTER ALLREDUCE (Element-wise Sum) ====");
    printAllRanks(rank, world_size, tensor.toString());

    // --- ReduceScatter ---
    int chunk_size = tensor.size() / world_size;
    float* recv_chunk;
    cudaMalloc(&recv_chunk, chunk_size * sizeof(float));
    
    // Create a send buffer with the all-reduced data
    pg.reduceScatter(recv_chunk, tensor.deviceData(), chunk_size, ncclFloat32)->wait();
    
    // Copy the received chunk to the beginning of our tensor
    cudaMemcpy(tensor.deviceData(), recv_chunk, chunk_size * sizeof(float), cudaMemcpyDeviceToDevice);
    tensor.copyDeviceToHost();
    
    printSectionHeader(rank, "==== AFTER REDUCE_SCATTER ====");
    printAllRanks(rank, world_size, tensor.toString());

    // --- AllGather ---
    float* gathered_data;
    cudaMalloc(&gathered_data, chunk_size * world_size * sizeof(float));
    
    pg.allGather(gathered_data, tensor.deviceData(), chunk_size, ncclFloat32)->wait();
    
    // Copy gathered data back
    cudaMemcpy(tensor.deviceData(), gathered_data, chunk_size * world_size * sizeof(float), cudaMemcpyDeviceToDevice);
    tensor.copyDeviceToHost();
    
    printSectionHeader(rank, "==== AFTER ALL_GATHER ====");
    printAllRanks(rank, world_size, tensor.toString());

    // --- Broadcast (root=0) ---
    pg.broadcast(tensor.deviceData(), tensor.size(), 0, ncclFloat32)->wait();
    tensor.copyDeviceToHost();
    
    printSectionHeader(rank, "==== AFTER BROADCAST (Root=0) ====");
    printAllRanks(rank, world_size, tensor.toString());

    // Cleanup
    cudaFree(recv_chunk);
    cudaFree(gathered_data);
}

// ---------------- Main ----------------
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // NCCL initialization
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Starting Distributed Tensor Example\n";
        std::cout << "World Size: " << world_size << " processes\n";
        std::cout << "Each using GPU corresponding to their rank\n";
    }

    worker(rank, world_size, id);

    if (rank == 0) {
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "All collectives completed successfully!\n";
    }

    MPI_Finalize();
    return 0;
}