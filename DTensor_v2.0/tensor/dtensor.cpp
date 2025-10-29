#include "dtensor.h"
#include <cuda_runtime.h>
#include <iostream>

DTensor::DTensor(int world_size, int slice_size, int rank, ProcessGroup* pg)
    : world_size_(world_size), slice_size_(slice_size), rank_(rank), pg_(pg) {
    h_data_.resize(slice_size_);
    for (int j = 0; j < slice_size_; ++j)
        h_data_[j] = float(rank_ * slice_size_ + j);
    cudaMalloc(&d_data_, slice_size_ * sizeof(float));
    cudaMemcpy(d_data_, h_data_.data(), slice_size_ * sizeof(float), cudaMemcpyHostToDevice);
}

DTensor::~DTensor() {
    if (d_data_) cudaFree(d_data_);
}

float* DTensor::deviceData() { return d_data_; }
std::vector<float>& DTensor::hostData() { return h_data_; }
size_t DTensor::size() const { return slice_size_; }

void DTensor::copyDeviceToHost() {
    cudaMemcpy(h_data_.data(), d_data_, slice_size_ * sizeof(float), cudaMemcpyDeviceToHost);
}

void DTensor::allReduce() {
    pg_->allReduce(d_data_, slice_size_, ncclFloat32)->wait();
}

void DTensor::reduceScatter() {
    int chunk_size = slice_size_ / world_size_;
    float* recv_chunk;
    cudaMalloc(&recv_chunk, chunk_size * sizeof(float));
    pg_->reduceScatter(recv_chunk, d_data_, chunk_size, ncclFloat32)->wait();
    cudaMemcpy(d_data_, recv_chunk, chunk_size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(recv_chunk);
}

void DTensor::allGather() {
    int chunk_size = slice_size_ / world_size_;
    float* gathered;
    cudaMalloc(&gathered, chunk_size * world_size_ * sizeof(float));
    pg_->allGather(gathered, d_data_, chunk_size, ncclFloat32)->wait();
    cudaMemcpy(d_data_, gathered, chunk_size * world_size_ * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(gathered);
}

void DTensor::broadcast() {
    pg_->broadcast(d_data_, slice_size_, 0, ncclFloat32)->wait();
}

std::string DTensor::toString() {
    std::stringstream ss;
    ss << "[Rank " << rank_ << "] ";
    for (float v : h_data_) ss << v << " ";
    return ss.str();
}
