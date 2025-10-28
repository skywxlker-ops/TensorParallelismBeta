#include "dtensor_mpi.hpp"
#include <iostream>
#include <sstream>

// ---------------- Work ----------------
Work::Work(cudaStream_t stream) : stream_(stream) {
    cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
}
Work::~Work() { cudaEventDestroy(event_); }

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
    : rank_(rank), world_size_(world_size), device_(device)
{
    cudaSetDevice(device_);
    cudaStreamCreate(&stream_);
    ncclCommInitRank(&comm_, world_size_, id, rank_);
}
ProcessGroup::~ProcessGroup() {
    ncclCommDestroy(comm_);
    cudaStreamDestroy(stream_);
}

// ---------------- DTensor ----------------
DTensor::DTensor(int world_size, int slice_size, int rank)
    : world_size_(world_size), slice_size_(slice_size), rank_(rank)
{
    h_data_.resize(slice_size_);
    for (int j = 0; j < slice_size_; ++j) {
        h_data_[j] = float(rank_ * slice_size_ + j);
    }
    cudaMalloc(&d_data_, slice_size_ * sizeof(float));
    cudaMemcpy(d_data_, h_data_.data(), slice_size_ * sizeof(float), cudaMemcpyHostToDevice);
}
DTensor::~DTensor() { if (d_data_) cudaFree(d_data_); }

float* DTensor::deviceData() { return d_data_; }
std::vector<float>& DTensor::hostData() { return h_data_; }
size_t DTensor::size() const { return static_cast<size_t>(slice_size_); }
void DTensor::copyDeviceToHost() {
    cudaMemcpy(h_data_.data(), d_data_, slice_size_ * sizeof(float), cudaMemcpyDeviceToHost);
}

// ---------------- Demo (excluded from pybind build) ----------------
//#ifndef DTENSOR_PYBIND_BUILD
// Put your original worker(...) and main(...) here.
//#endif
