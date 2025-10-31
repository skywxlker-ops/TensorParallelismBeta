// #include "dtensor.h"
// #include <cuda_runtime.h>
// #include <nccl.h>
// #include <fstream>
// #include <iostream>

// DTensor::DTensor(int rank, int world_size, ProcessGroup* pg)
//     : rank_(rank), world_size_(world_size), pg_(pg) {
//     size_ = 8;
//     shape_[0] = size_;
//     cudaSetDevice(rank_);
//     cudaMalloc(&data_, size_ * sizeof(float));
//     cudaMalloc(&temp_buf_, size_ * world_size_ * sizeof(float));

//     std::vector<float> host_data(size_);
//     for (int i = 0; i < size_; i++) {
//         host_data[i] = static_cast<float>(rank_ * size_ + i);
//     }
//     cudaMemcpy(data_, host_data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
// }

// DTensor::~DTensor() {
//     cudaFree(data_);
//     cudaFree(temp_buf_);
// }

// // ------------------- EXISTING METHODS -------------------

// void DTensor::setData(const std::vector<float>& host_data) {
//     size_ = host_data.size();
//     shape_[0] = size_;
//     cudaMemcpy(data_, host_data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
// }

// std::vector<float> DTensor::getData() const {
//     std::vector<float> host_data(size_);
//     cudaMemcpy(host_data.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
//     return host_data;
// }

// void DTensor::allReduce() {
//     pg_->allReduce<float>(data_, size_, ncclFloat);
//     cudaDeviceSynchronize();
// }

// void DTensor::reduceScatter() {
//     pg_->reduceScatter<float>(data_, temp_buf_, size_, ncclFloat);
//     cudaDeviceSynchronize();
// }

// void DTensor::allGather() {
//     pg_->allGather<float>(temp_buf_, data_, size_, ncclFloat);
//     cudaDeviceSynchronize();
// }

// void DTensor::broadcast(int root) {
//     pg_->broadcast<float>(data_, size_, root, ncclFloat);
//     cudaDeviceSynchronize();
// }

// void DTensor::print() const {
//     std::vector<float> host_data(size_);
//     cudaMemcpy(host_data.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
//     std::cout << "[Rank " << rank_ << "] ";
//     for (auto x : host_data) std::cout << x << " ";
//     std::cout << std::endl;
// }

// // ------------------- NEW CHECKPOINT METHODS -------------------

// void DTensor::saveCheckpoint(const std::string& path) const {
//     std::vector<float> host_data(size_);
//     cudaMemcpy(host_data.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);

//     std::ofstream file(path, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for writing: " << path << std::endl;
//         return;
//     }

//     file.write((char*)&size_, sizeof(int));
//     file.write((char*)&shape_[0], sizeof(int));
//     file.write(dtype_.c_str(), dtype_.size() + 1);
//     file.write((char*)host_data.data(), size_ * sizeof(float));
//     file.close();

//     std::cout << "[Rank " << rank_ << "] Checkpoint saved to " << path << std::endl;
// }

// void DTensor::loadCheckpoint(const std::string& path) {
//     std::ifstream file(path, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for reading: " << path << std::endl;
//         return;
//     }

//     int saved_size;
//     int saved_shape;
//     char dtype_buf[32];

//     file.read((char*)&saved_size, sizeof(int));
//     file.read((char*)&saved_shape, sizeof(int));
//     file.read(dtype_buf, sizeof(dtype_buf));
//     dtype_ = std::string(dtype_buf);

//     std::vector<float> host_data(saved_size);
//     file.read((char*)host_data.data(), saved_size * sizeof(float));
//     file.close();

//     if (saved_size != size_) {
//         cudaFree(data_);
//         size_ = saved_size;
//         shape_[0] = saved_shape;
//         cudaMalloc(&data_, size_ * sizeof(float));
//     }

//     cudaMemcpy(data_, host_data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
//     std::cout << "[Rank " << rank_ << "] Checkpoint loaded from " << path << std::endl;
// }



#include "dtensor.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <fstream>
#include <iostream>

// ---------------- GLOBAL ALLOCATOR INSTANCE ----------------
CachingAllocator gAllocator;

// ---------------- CONSTRUCTOR / DESTRUCTOR ----------------

DTensor::DTensor(int rank, int world_size, ProcessGroup* pg)
    : rank_(rank), world_size_(world_size), pg_(pg),
      data_(nullptr), temp_buf_(nullptr), data_block_(nullptr), temp_block_(nullptr) {

    size_ = 8;
    shape_[0] = size_;

    cudaSetDevice(rank_);
    cudaStreamCreate(&stream_);

    // Allocate using our stream-aware caching allocator
    data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
    temp_block_ = gAllocator.allocateMemory(size_ * world_size_ * sizeof(float), stream_);

    data_ = static_cast<float*>(data_block_->addr);
    temp_buf_ = static_cast<float*>(temp_block_->addr);

    std::vector<float> host_data(size_);
    for (int i = 0; i < size_; i++) {
        host_data[i] = static_cast<float>(rank_ * size_ + i);
    }

    cudaMemcpyAsync(data_, host_data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);
}

DTensor::~DTensor() {
    cudaStreamSynchronize(stream_);
    if (data_block_) gAllocator.freeMemory(data_block_);
    if (temp_block_) gAllocator.freeMemory(temp_block_);
    cudaStreamDestroy(stream_);
}

// ---------------- EXISTING METHODS ----------------

void DTensor::setData(const std::vector<float>& host_data) {
    size_ = host_data.size();
    shape_[0] = size_;
    cudaMemcpyAsync(data_, host_data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);
}

std::vector<float> DTensor::getData() const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return host_data;
}

void DTensor::allReduce() {
    pg_->allReduce<float>(data_, size_, ncclFloat);
    cudaStreamSynchronize(stream_);
}

void DTensor::reduceScatter() {
    pg_->reduceScatter<float>(data_, temp_buf_, size_, ncclFloat);
    cudaStreamSynchronize(stream_);
}

void DTensor::allGather() {
    pg_->allGather<float>(temp_buf_, data_, size_, ncclFloat);
    cudaStreamSynchronize(stream_);
}

void DTensor::broadcast(int root) {
    pg_->broadcast<float>(data_, size_, root, ncclFloat);
    cudaStreamSynchronize(stream_);
}

void DTensor::print() const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    std::cout << "[Rank " << rank_ << "] ";
    for (auto x : host_data) std::cout << x << " ";
    std::cout << std::endl;
}

// ---------------- CHECKPOINT METHODS ----------------

void DTensor::saveCheckpoint(const std::string& path) const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for writing: " << path << std::endl;
        return;
    }

    file.write((char*)&size_, sizeof(int));
    file.write((char*)&shape_[0], sizeof(int));
    file.write(dtype_.c_str(), dtype_.size() + 1);
    file.write((char*)host_data.data(), size_ * sizeof(float));
    file.close();

    std::cout << "[Rank " << rank_ << "] Checkpoint saved to " << path << std::endl;
}

void DTensor::loadCheckpoint(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for reading: " << path << std::endl;
        return;
    }

    int saved_size;
    int saved_shape;
    char dtype_buf[32];

    file.read((char*)&saved_size, sizeof(int));
    file.read((char*)&saved_shape, sizeof(int));
    file.read(dtype_buf, sizeof(dtype_buf));
    dtype_ = std::string(dtype_buf);

    std::vector<float> host_data(saved_size);
    file.read((char*)host_data.data(), saved_size * sizeof(float));
    file.close();

    if (saved_size != size_) {
        gAllocator.freeMemory(data_block_);
        data_block_ = gAllocator.allocateMemory(saved_size * sizeof(float), stream_);
        data_ = static_cast<float*>(data_block_->addr);

        size_ = saved_size;
        shape_[0] = saved_shape;
    }

    cudaMemcpyAsync(data_, host_data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    std::cout << "[Rank " << rank_ << "] Checkpoint loaded from " << path << std::endl;
}
