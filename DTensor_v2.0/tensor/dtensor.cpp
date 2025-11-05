// #include "dtensor.h"
// #include <cuda_runtime.h>
// #include <nccl.h>
// #include <fstream>
// #include <iostream>

// CachingAllocator gAllocator;

// // =========================================================
// // DTensor: TensorLib + NCCL integration
// // =========================================================
// DTensor::DTensor(int rank, int world_size, ProcessGroup* pg)
//     : rank_(rank),
//       world_size_(world_size),
//       pg_(pg),
//       data_block_(nullptr),
//       temp_block_(nullptr),
//       size_(0),
//       //  Properly initialize TensorLib tensors
//       tensor_(OwnTensor::Shape{{1}},
//               OwnTensor::TensorOptions()
//                   .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
//                   .with_dtype(OwnTensor::Dtype::Float32)),
//       temp_tensor_(OwnTensor::Shape{{1}},
//                    OwnTensor::TensorOptions()
//                        .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
//                        .with_dtype(OwnTensor::Dtype::Float32))
// {
//     cudaSetDevice(rank_);
//     stream_ = pg_->getStream();
// }

// DTensor::~DTensor() {
//     cudaStreamSynchronize(stream_);
//     if (data_block_) gAllocator.freeMemory(data_block_);
//     if (temp_block_) gAllocator.freeMemory(temp_block_);
// }

// // =========================================================
// // Setup & Data Transfer (TensorLib-backed tensors)
// // =========================================================
// void DTensor::setData(const std::vector<float>& host_data) {
//     size_ = static_cast<int>(host_data.size());
//     shape_[0] = size_;
//     dtype_ = "float32";

//     // --- Create GPU tensors ---
//     OwnTensor::Shape shape{{size_}};
//     OwnTensor::TensorOptions opts;
//     opts = opts
//         .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
//         .with_dtype(OwnTensor::Dtype::Float32);

//     tensor_ = OwnTensor::Tensor(shape, opts);
//     temp_tensor_ = OwnTensor::Tensor(shape, opts);


//     tensor_.set_data(host_data);

//     // Keep legacy allocator for monitoring memory behavior
//     if (data_block_) gAllocator.freeMemory(data_block_);
//     if (temp_block_) gAllocator.freeMemory(temp_block_);
//     data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
//     temp_block_ = gAllocator.allocateMemory(size_ * world_size_ * sizeof(float), stream_);
// }

// std::vector<float> DTensor::getData() const {
//     std::vector<float> host_data(size_);
//     cudaMemcpyAsync(host_data.data(),
//                     tensor_.data<float>(),
//                     size_ * sizeof(float),
//                     cudaMemcpyDeviceToHost,
//                     stream_);
//     cudaStreamSynchronize(stream_);
//     return host_data;
// }

// // =========================================================
// // NCCL GPU Collectives
// // =========================================================
// void DTensor::allReduce() {
//     auto work = pg_->allReduce<float>(tensor_.data<float>(), size_, ncclFloat);
//     work->wait();
// }

// void DTensor::reduceScatter() {
//     auto work = pg_->reduceScatter<float>(
//         temp_tensor_.data<float>(), tensor_.data<float>(), size_, ncclFloat);
//     work->wait();
// }

// void DTensor::allGather() {
//     auto work = pg_->allGather<float>(
//         temp_tensor_.data<float>(), tensor_.data<float>(), size_, ncclFloat);
//     work->wait();
// }

// void DTensor::broadcast(int root) {
//     auto work = pg_->broadcast<float>(tensor_.data<float>(), size_, root, ncclFloat);
//     work->wait();
// }

// // =========================================================
// // Printing Utility
// // =========================================================
// void DTensor::print() const {
//     std::vector<float> host_data(size_);
//     cudaMemcpyAsync(host_data.data(),
//                     tensor_.data<float>(),
//                     size_ * sizeof(float),
//                     cudaMemcpyDeviceToHost,
//                     stream_);
//     cudaStreamSynchronize(stream_);

//     std::cout << "[Rank " << rank_ << "] ";
//     for (int i = 0; i < std::min(size_, 10); ++i)
//         std::cout << host_data[i] << " ";
//     if (size_ > 10) std::cout << "...";
//     std::cout << std::endl;
// }

// // =========================================================
// // Checkpointing: GPU ↔ Disk
// // =========================================================
// void DTensor::saveCheckpoint(const std::string& path) const {
//     std::vector<float> host_data(size_);
//     cudaMemcpyAsync(host_data.data(),
//                     tensor_.data<float>(),
//                     size_ * sizeof(float),
//                     cudaMemcpyDeviceToHost,
//                     stream_);
//     cudaStreamSynchronize(stream_);

//     std::ofstream file(path, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for writing: "
//                   << path << std::endl;
//         return;
//     }

//     file.write((char*)&size_, sizeof(int));
//     file.write((char*)&shape_[0], sizeof(int));
//     file.write(dtype_.c_str(), dtype_.size() + 1);
//     file.write(reinterpret_cast<char*>(host_data.data()), size_ * sizeof(float));
//     file.close();

//     std::cout << "[Rank " << rank_ << "] Checkpoint saved to " << path << std::endl;
// }

// void DTensor::loadCheckpoint(const std::string& path) {
//     std::ifstream file(path, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for reading: "
//                   << path << std::endl;
//         return;
//     }

//     int saved_size;
//     int saved_shape;
//     char dtype_buf[32];

//     file.read(reinterpret_cast<char*>(&saved_size), sizeof(int));
//     file.read(reinterpret_cast<char*>(&saved_shape), sizeof(int));
//     file.read(dtype_buf, sizeof(dtype_buf));
//     dtype_ = std::string(dtype_buf);

//     std::vector<float> host_data(saved_size);
//     file.read(reinterpret_cast<char*>(host_data.data()), saved_size * sizeof(float));
//     file.close();

//     // Rebuild tensor if size changed
//     if (saved_size != size_) {
//         OwnTensor::Shape shape{{saved_size}};
//         OwnTensor::TensorOptions opts;
//         opts = opts
//             .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
//             .with_dtype(OwnTensor::Dtype::Float32);
//         tensor_ = OwnTensor::Tensor(shape, opts);

//         size_ = saved_size;
//         shape_[0] = saved_shape;
//     }

//     tensor_.set_data(host_data);
//     std::cout << "[Rank " << rank_ << "] Checkpoint loaded from " << path << std::endl;
// }

#include "tensor/dtensor.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <fstream>
#include <iostream>
#include <filesystem>

CachingAllocator gAllocator;

// =========================================================
// DTensor Constructor/Destructor
// =========================================================
DTensor::DTensor(int rank, int world_size, ProcessGroup* pg)
    : rank_(rank),
      world_size_(world_size),
      size_(0),
      pg_(pg),
      stream_(nullptr),
      data_block_(nullptr),
      temp_block_(nullptr),
      tensor_(OwnTensor::Shape{{1}},
              OwnTensor::TensorOptions()
                  .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
                  .with_dtype(OwnTensor::Dtype::Float32)),
      temp_tensor_(OwnTensor::Shape{{1}},
                   OwnTensor::TensorOptions()
                       .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
                       .with_dtype(OwnTensor::Dtype::Float32)) {
    cudaSetDevice(rank_);
    stream_ = pg_->getStream();
}

DTensor::~DTensor() {
    cudaStreamSynchronize(stream_);
    if (data_block_) gAllocator.freeMemory(data_block_);
    if (temp_block_) gAllocator.freeMemory(temp_block_);
}

// =========================================================
// Setup & Data Transfer
// =========================================================
void DTensor::setData(const std::vector<float>& host_data) {
    size_ = static_cast<int>(host_data.size());
    shape_[0] = size_;
    dtype_ = "float32";

    OwnTensor::Shape shape{{size_}};
    OwnTensor::TensorOptions opts;
    opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
               .with_dtype(OwnTensor::Dtype::Float32);

    tensor_ = OwnTensor::Tensor(shape, opts);
    temp_tensor_ = OwnTensor::Tensor(shape, opts);
    tensor_.set_data(host_data);

    if (data_block_) gAllocator.freeMemory(data_block_);
    if (temp_block_) gAllocator.freeMemory(temp_block_);
    data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
    temp_block_ = gAllocator.allocateMemory(size_ * world_size_ * sizeof(float), stream_);
}

std::vector<float> DTensor::getData() const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return host_data;
}

// =========================================================
// Collectives
// =========================================================
void DTensor::allReduce() {
    auto work = pg_->allReduce<float>(tensor_.data<float>(), size_, ncclFloat);
    work->wait();
}

void DTensor::reduceScatter() {
    auto work = pg_->reduceScatter<float>(
        temp_tensor_.data<float>(), tensor_.data<float>(), size_, ncclFloat);
    work->wait();
}

void DTensor::allGather() {
    auto work = pg_->allGather<float>(
        temp_tensor_.data<float>(), tensor_.data<float>(), size_, ncclFloat);
    work->wait();
}

void DTensor::broadcast(int root) {
    auto work = pg_->broadcast<float>(tensor_.data<float>(), size_, root, ncclFloat);
    work->wait();
}

// =========================================================
// TensorOps (Bridge Integration)
// =========================================================
DTensor DTensor::add(const DTensor& other) const {
    OwnTensor::Tensor result = TensorOpsBridge::add(tensor_, other.tensor_);
    DTensor out(rank_, world_size_, pg_);
    out.tensor_ = result;
    out.dtype_ = "float32";

    out.size_ = 1;
    for (auto d : result.shape().dims) out.size_ *= d;
    out.shape_[0] = out.size_;
    return out;
}

DTensor DTensor::sub(const DTensor& other) const {
    OwnTensor::Tensor result = TensorOpsBridge::sub(tensor_, other.tensor_);
    DTensor out(rank_, world_size_, pg_);
    out.tensor_ = result;
    out.dtype_ = "float32";

    out.size_ = 1;
    for (auto d : result.shape().dims) out.size_ *= d;
    out.shape_[0] = out.size_;
    return out;
}

DTensor DTensor::mul(const DTensor& other) const {
    OwnTensor::Tensor result = TensorOpsBridge::mul(tensor_, other.tensor_);
    DTensor out(rank_, world_size_, pg_);
    out.tensor_ = result;
    out.dtype_ = "float32";

    out.size_ = 1;
    for (auto d : result.shape().dims) out.size_ *= d;
    out.shape_[0] = out.size_;
    return out;
}

DTensor DTensor::div(const DTensor& other) const {
    OwnTensor::Tensor result = TensorOpsBridge::div(tensor_, other.tensor_);
    DTensor out(rank_, world_size_, pg_);
    out.tensor_ = result;
    out.dtype_ = "float32";

    out.size_ = 1;
    for (auto d : result.shape().dims) out.size_ *= d;
    out.shape_[0] = out.size_;
    return out;
}

DTensor DTensor::matmul(const DTensor& other) const {
    OwnTensor::Tensor result = TensorOpsBridge::matmul(tensor_, other.tensor_);
    DTensor out(rank_, world_size_, pg_);
    out.tensor_ = result;
    out.dtype_ = "float32";

    out.size_ = 1;
    for (auto d : result.shape().dims) out.size_ *= d;
    out.shape_[0] = out.size_;
    return out;
}

DTensor DTensor::reshape(int rows, int cols) const {
    if (rows * cols != size_) {
        throw std::runtime_error("DTensor::reshape: total elements mismatch");
    }

    OwnTensor::Shape new_shape{{rows, cols}};
    OwnTensor::Tensor reshaped_tensor = tensor_.reshape(new_shape);

    DTensor out(rank_, world_size_, pg_);
    out.tensor_ = reshaped_tensor;
    out.size_ = size_;
    out.dtype_ = "float32";
    out.shape_[0] = size_;
    return out;
}

// =========================================================
// Checkpointing
// =========================================================
void DTensor::saveCheckpoint(const std::string& path) const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for writing: "
                  << path << std::endl;
        return;
    }

    file.write((char*)&size_, sizeof(int));
    file.write((char*)&shape_[0], sizeof(int));
    file.write(dtype_.c_str(), dtype_.size() + 1);
    file.write(reinterpret_cast<char*>(host_data.data()), size_ * sizeof(float));
    file.close();
    std::cout << "[Rank " << rank_ << "] Checkpoint saved to " << path << std::endl;
}

void DTensor::loadCheckpoint(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for reading: "
                  << path << std::endl;
        return;
    }

    int saved_size, saved_shape;
    char dtype_buf[32];
    file.read(reinterpret_cast<char*>(&saved_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_shape), sizeof(int));
    file.read(dtype_buf, sizeof(dtype_buf));
    dtype_ = std::string(dtype_buf);

    std::vector<float> host_data(saved_size);
    file.read(reinterpret_cast<char*>(host_data.data()), saved_size * sizeof(float));
    file.close();

    if (saved_size != size_) {
        OwnTensor::Shape shape{{saved_size}};
        OwnTensor::TensorOptions opts;
        opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
                   .with_dtype(OwnTensor::Dtype::Float32);
        tensor_ = OwnTensor::Tensor(shape, opts);
        size_ = saved_size;
        shape_[0] = saved_shape;
    }

    tensor_.set_data(host_data);
    std::cout << "[Rank " << rank_ << "] Checkpoint loaded from " << path << std::endl;
}

// =========================================================
// Debug Printing
// =========================================================
void DTensor::print() const {
    if (size_ <= 0 || tensor_.data<float>() == nullptr) {
        std::cerr << "[Rank " << rank_ << "] Tensor is empty or uninitialized!\n";
        return;
    }

    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(),
                    tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronize(stream_);

    auto s = tensor_.shape().dims;
    std::cout << "[Rank " << rank_ << "] ";

    if (s.size() == 1) {
        for (int i = 0; i < std::min(size_, 10); ++i)
            std::cout << host_data[i] << " ";
    } else if (s.size() == 2) {
        int rows = s[0], cols = s[1];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                std::cout << host_data[i * cols + j] << " ";
            std::cout << "\n[Rank " << rank_ << "] ";
        }
    } else {
        std::cout << "(Unsupported print shape: ";
        for (auto d : s) std::cout << d << " ";
        std::cout << ")";
    }

    std::cout << std::endl;
}
