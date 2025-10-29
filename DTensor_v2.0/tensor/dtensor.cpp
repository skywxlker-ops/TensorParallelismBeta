#include "dtensor.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>

DTensor::DTensor(int rank, int world_size, ProcessGroup* pg)
    : rank_(rank), world_size_(world_size), pg_(pg) {
    size_ = 8;
    cudaSetDevice(rank_);
    cudaMalloc(&data_, size_ * sizeof(float));
    cudaMalloc(&temp_buf_, size_ * world_size_ * sizeof(float));

    std::vector<float> host_data(size_);
    for (int i = 0; i < size_; i++) {
        host_data[i] = static_cast<float>(rank_ * size_ + i);
    }
    cudaMemcpy(data_, host_data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
}

DTensor::~DTensor() {
    cudaFree(data_);
    cudaFree(temp_buf_);
}

// ------------------- NEW METHODS -------------------

void DTensor::setData(const std::vector<float>& host_data) {
    size_ = host_data.size();
    cudaMemcpy(data_, host_data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
}

std::vector<float> DTensor::getData() const {
    std::vector<float> host_data(size_);
    cudaMemcpy(host_data.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    return host_data;
}

// ---------------------------------------------------

void DTensor::allReduce() {
    pg_->allReduce<float>(data_, size_, ncclFloat);
    cudaDeviceSynchronize();
}

void DTensor::reduceScatter() {
    pg_->reduceScatter<float>(data_, temp_buf_, size_, ncclFloat);
    cudaDeviceSynchronize();
}

void DTensor::allGather() {
    pg_->allGather<float>(temp_buf_, data_, size_, ncclFloat);
    cudaDeviceSynchronize();
}

void DTensor::broadcast(int root) {
    pg_->broadcast<float>(data_, size_, root, ncclFloat);
    cudaDeviceSynchronize();
}

void DTensor::print() const {
    std::vector<float> host_data(size_);
    cudaMemcpy(host_data.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[Rank " << rank_ << "] ";
    for (auto x : host_data) std::cout << x << " ";
    std::cout << std::endl;
}
