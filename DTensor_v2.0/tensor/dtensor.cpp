// // #include "dtensor.h"
// // #include <cuda_runtime.h>
// // #include <nccl.h>
// // #include <fstream>
// // #include <iostream>

// // CachingAllocator gAllocator;

// // // =========================================================
// // // DTensor: TensorLib + NCCL integration
// // // =========================================================
// // DTensor::DTensor(int rank, int world_size, ProcessGroup* pg)
// //     : rank_(rank),
// //       world_size_(world_size),
// //       pg_(pg),
// //       data_block_(nullptr),
// //       temp_block_(nullptr),
// //       size_(0),
// //       //  Properly initialize TensorLib tensors
// //       tensor_(OwnTensor::Shape{{1}},
// //               OwnTensor::TensorOptions()
// //                   .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
// //                   .with_dtype(OwnTensor::Dtype::Float32)),
// //       temp_tensor_(OwnTensor::Shape{{1}},
// //                    OwnTensor::TensorOptions()
// //                        .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
// //                        .with_dtype(OwnTensor::Dtype::Float32))
// // {
// //     cudaSetDevice(rank_);
// //     stream_ = pg_->getStream();
// // }

// // DTensor::~DTensor() {
// //     cudaStreamSynchronize(stream_);
// //     if (data_block_) gAllocator.freeMemory(data_block_);
// //     if (temp_block_) gAllocator.freeMemory(temp_block_);
// // }

// // // =========================================================
// // // Setup & Data Transfer (TensorLib-backed tensors)
// // // =========================================================
// // void DTensor::setData(const std::vector<float>& host_data) {
// //     size_ = static_cast<int>(host_data.size());
// //     shape_[0] = size_;
// //     dtype_ = "float32";

// //     // --- Create GPU tensors ---
// //     OwnTensor::Shape shape{{size_}};
// //     OwnTensor::TensorOptions opts;
// //     opts = opts
// //         .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
// //         .with_dtype(OwnTensor::Dtype::Float32);

// //     tensor_ = OwnTensor::Tensor(shape, opts);
// //     temp_tensor_ = OwnTensor::Tensor(shape, opts);


// //     tensor_.set_data(host_data);

// //     // Keep legacy allocator for monitoring memory behavior
// //     if (data_block_) gAllocator.freeMemory(data_block_);
// //     if (temp_block_) gAllocator.freeMemory(temp_block_);
// //     data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
// //     temp_block_ = gAllocator.allocateMemory(size_ * world_size_ * sizeof(float), stream_);
// // }

// // std::vector<float> DTensor::getData() const {
// //     std::vector<float> host_data(size_);
// //     cudaMemcpyAsync(host_data.data(),
// //                     tensor_.data<float>(),
// //                     size_ * sizeof(float),
// //                     cudaMemcpyDeviceToHost,
// //                     stream_);
// //     cudaStreamSynchronize(stream_);
// //     return host_data;
// // }

// // // =========================================================
// // // NCCL GPU Collectives
// // // =========================================================
// // void DTensor::allReduce() {
// //     auto work = pg_->allReduce<float>(tensor_.data<float>(), size_, ncclFloat);
// //     work->wait();
// // }

// // void DTensor::reduceScatter() {
// //     auto work = pg_->reduceScatter<float>(
// //         temp_tensor_.data<float>(), tensor_.data<float>(), size_, ncclFloat);
// //     work->wait();
// // }

// // void DTensor::allGather() {
// //     auto work = pg_->allGather<float>(
// //         temp_tensor_.data<float>(), tensor_.data<float>(), size_, ncclFloat);
// //     work->wait();
// // }

// // void DTensor::broadcast(int root) {
// //     auto work = pg_->broadcast<float>(tensor_.data<float>(), size_, root, ncclFloat);
// //     work->wait();
// // }

// // // =========================================================
// // // Printing Utility
// // // =========================================================
// // void DTensor::print() const {
// //     std::vector<float> host_data(size_);
// //     cudaMemcpyAsync(host_data.data(),
// //                     tensor_.data<float>(),
// //                     size_ * sizeof(float),
// //                     cudaMemcpyDeviceToHost,
// //                     stream_);
// //     cudaStreamSynchronize(stream_);

// //     std::cout << "[Rank " << rank_ << "] ";
// //     for (int i = 0; i < std::min(size_, 10); ++i)
// //         std::cout << host_data[i] << " ";
// //     if (size_ > 10) std::cout << "...";
// //     std::cout << std::endl;
// // }

// // // =========================================================
// // // Checkpointing: GPU ↔ Disk
// // // =========================================================
// // void DTensor::saveCheckpoint(const std::string& path) const {
// //     std::vector<float> host_data(size_);
// //     cudaMemcpyAsync(host_data.data(),
// //                     tensor_.data<float>(),
// //                     size_ * sizeof(float),
// //                     cudaMemcpyDeviceToHost,
// //                     stream_);
// //     cudaStreamSynchronize(stream_);

// //     std::ofstream file(path, std::ios::binary);
// //     if (!file.is_open()) {
// //         std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for writing: "
// //                   << path << std::endl;
// //         return;
// //     }

// //     file.write((char*)&size_, sizeof(int));
// //     file.write((char*)&shape_[0], sizeof(int));
// //     file.write(dtype_.c_str(), dtype_.size() + 1);
// //     file.write(reinterpret_cast<char*>(host_data.data()), size_ * sizeof(float));
// //     file.close();

// //     std::cout << "[Rank " << rank_ << "] Checkpoint saved to " << path << std::endl;
// // }

// // void DTensor::loadCheckpoint(const std::string& path) {
// //     std::ifstream file(path, std::ios::binary);
// //     if (!file.is_open()) {
// //         std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for reading: "
// //                   << path << std::endl;
// //         return;
// //     }

// //     int saved_size;
// //     int saved_shape;
// //     char dtype_buf[32];

// //     file.read(reinterpret_cast<char*>(&saved_size), sizeof(int));
// //     file.read(reinterpret_cast<char*>(&saved_shape), sizeof(int));
// //     file.read(dtype_buf, sizeof(dtype_buf));
// //     dtype_ = std::string(dtype_buf);

// //     std::vector<float> host_data(saved_size);
// //     file.read(reinterpret_cast<char*>(host_data.data()), saved_size * sizeof(float));
// //     file.close();

// //     // Rebuild tensor if size changed
// //     if (saved_size != size_) {
// //         OwnTensor::Shape shape{{saved_size}};
// //         OwnTensor::TensorOptions opts;
// //         opts = opts
// //             .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
// //             .with_dtype(OwnTensor::Dtype::Float32);
// //         tensor_ = OwnTensor::Tensor(shape, opts);

// //         size_ = saved_size;
// //         shape_[0] = saved_shape;
// //     }

// //     tensor_.set_data(host_data);
// //     std::cout << "[Rank " << rank_ << "] Checkpoint loaded from " << path << std::endl;
// // }

// #include "tensor/dtensor.h"
// #include <cuda_runtime.h>
// #include <nccl.h>
// #include <fstream>
// #include <iostream>
// #include <filesystem>

// CachingAllocator gAllocator;

// // =========================================================
// // DTensor Constructor/Destructor
// // =========================================================
// DTensor::DTensor(int rank, int world_size, ProcessGroup* pg)
//     : rank_(rank),
//       world_size_(world_size),
//       size_(0),
//       pg_(pg),
//       stream_(nullptr),
//       data_block_(nullptr),
//       temp_block_(nullptr),
//       tensor_(OwnTensor::Shape{{1}},
//               OwnTensor::TensorOptions()
//                   .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
//                   .with_dtype(OwnTensor::Dtype::Float32)),
//       temp_tensor_(OwnTensor::Shape{{1}},
//                    OwnTensor::TensorOptions()
//                        .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
//                        .with_dtype(OwnTensor::Dtype::Float32)) {
//     cudaSetDevice(rank_);
//     stream_ = pg_->getStream();
// }

// DTensor::~DTensor() {
//     cudaStreamSynchronize(stream_);
//     if (data_block_) gAllocator.freeMemory(data_block_);
//     if (temp_block_) gAllocator.freeMemory(temp_block_);
// }

// // =========================================================
// // Setup & Data Transfer
// // =========================================================
// void DTensor::setData(const std::vector<float>& host_data) {
//     size_ = static_cast<int>(host_data.size());
//     shape_[0] = size_;
//     dtype_ = "float32";

//     OwnTensor::Shape shape{{size_}};
//     OwnTensor::TensorOptions opts;
//     opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
//                .with_dtype(OwnTensor::Dtype::Float32);

//     tensor_ = OwnTensor::Tensor(shape, opts);
//     temp_tensor_ = OwnTensor::Tensor(shape, opts);
//     tensor_.set_data(host_data);

//     if (data_block_) gAllocator.freeMemory(data_block_);
//     if (temp_block_) gAllocator.freeMemory(temp_block_);
//     data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
//     temp_block_ = gAllocator.allocateMemory(size_ * world_size_ * sizeof(float), stream_);
// }

// std::vector<float> DTensor::getData() const {
//     std::vector<float> host_data(size_);
//     cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
//                     size_ * sizeof(float),
//                     cudaMemcpyDeviceToHost, stream_);
//     cudaStreamSynchronize(stream_);
//     return host_data;
// }

// // =========================================================
// // Collectives
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
// // TensorOps (Bridge Integration)
// // =========================================================
// DTensor DTensor::add(const DTensor& other) const {
//     OwnTensor::Tensor result = TensorOpsBridge::add(tensor_, other.tensor_);
//     DTensor out(rank_, world_size_, pg_);
//     out.tensor_ = result;
//     out.dtype_ = "float32";

//     out.size_ = 1;
//     for (auto d : result.shape().dims) out.size_ *= d;
//     out.shape_[0] = out.size_;
//     return out;
// }

// DTensor DTensor::sub(const DTensor& other) const {
//     OwnTensor::Tensor result = TensorOpsBridge::sub(tensor_, other.tensor_);
//     DTensor out(rank_, world_size_, pg_);
//     out.tensor_ = result;
//     out.dtype_ = "float32";

//     out.size_ = 1;
//     for (auto d : result.shape().dims) out.size_ *= d;
//     out.shape_[0] = out.size_;
//     return out;
// }

// DTensor DTensor::mul(const DTensor& other) const {
//     OwnTensor::Tensor result = TensorOpsBridge::mul(tensor_, other.tensor_);
//     DTensor out(rank_, world_size_, pg_);
//     out.tensor_ = result;
//     out.dtype_ = "float32";

//     out.size_ = 1;
//     for (auto d : result.shape().dims) out.size_ *= d;
//     out.shape_[0] = out.size_;
//     return out;
// }

// DTensor DTensor::div(const DTensor& other) const {
//     OwnTensor::Tensor result = TensorOpsBridge::div(tensor_, other.tensor_);
//     DTensor out(rank_, world_size_, pg_);
//     out.tensor_ = result;
//     out.dtype_ = "float32";

//     out.size_ = 1;
//     for (auto d : result.shape().dims) out.size_ *= d;
//     out.shape_[0] = out.size_;
//     return out;
// }

// DTensor DTensor::matmul(const DTensor& other) const {
//     OwnTensor::Tensor result = TensorOpsBridge::matmul(tensor_, other.tensor_);
//     DTensor out(rank_, world_size_, pg_);
//     out.tensor_ = result;
//     out.dtype_ = "float32";

//     out.size_ = 1;
//     for (auto d : result.shape().dims) out.size_ *= d;
//     out.shape_[0] = out.size_;
//     return out;
// }

// DTensor DTensor::reshape(int rows, int cols) const {
//     if (rows * cols != size_) {
//         throw std::runtime_error("DTensor::reshape: total elements mismatch");
//     }

//     OwnTensor::Shape new_shape{{rows, cols}};
//     OwnTensor::Tensor reshaped_tensor = tensor_.reshape(new_shape);

//     DTensor out(rank_, world_size_, pg_);
//     out.tensor_ = reshaped_tensor;
//     out.size_ = size_;
//     out.dtype_ = "float32";
//     out.shape_[0] = size_;
//     return out;
// }

// // =========================================================
// // Checkpointing
// // =========================================================
// void DTensor::saveCheckpoint(const std::string& path) const {
//     std::vector<float> host_data(size_);
//     cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
//                     size_ * sizeof(float),
//                     cudaMemcpyDeviceToHost, stream_);
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

//     int saved_size, saved_shape;
//     char dtype_buf[32];
//     file.read(reinterpret_cast<char*>(&saved_size), sizeof(int));
//     file.read(reinterpret_cast<char*>(&saved_shape), sizeof(int));
//     file.read(dtype_buf, sizeof(dtype_buf));
//     dtype_ = std::string(dtype_buf);

//     std::vector<float> host_data(saved_size);
//     file.read(reinterpret_cast<char*>(host_data.data()), saved_size * sizeof(float));
//     file.close();

//     if (saved_size != size_) {
//         OwnTensor::Shape shape{{saved_size}};
//         OwnTensor::TensorOptions opts;
//         opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
//                    .with_dtype(OwnTensor::Dtype::Float32);
//         tensor_ = OwnTensor::Tensor(shape, opts);
//         size_ = saved_size;
//         shape_[0] = saved_shape;
//     }

//     tensor_.set_data(host_data);
//     std::cout << "[Rank " << rank_ << "] Checkpoint loaded from " << path << std::endl;
// }

// // =========================================================
// // Debug Printing
// // =========================================================
// void DTensor::print() const {
//     if (size_ <= 0 || tensor_.data<float>() == nullptr) {
//         std::cerr << "[Rank " << rank_ << "] Tensor is empty or uninitialized!\n";
//         return;
//     }

//     std::vector<float> host_data(size_);
//     cudaMemcpyAsync(host_data.data(),
//                     tensor_.data<float>(),
//                     size_ * sizeof(float),
//                     cudaMemcpyDeviceToHost,
//                     stream_);
//     cudaStreamSynchronize(stream_);

//     auto s = tensor_.shape().dims;
//     std::cout << "[Rank " << rank_ << "] ";

//     if (s.size() == 1) {
//         for (int i = 0; i < std::min(size_, 10); ++i)
//             std::cout << host_data[i] << " ";
//     } else if (s.size() == 2) {
//         int rows = s[0], cols = s[1];
//         for (int i = 0; i < rows; ++i) {
//             for (int j = 0; j < cols; ++j)
//                 std::cout << host_data[i * cols + j] << " ";
//             std::cout << "\n[Rank " << rank_ << "] ";
//         }
//     } else {
//         std::cout << "(Unsupported print shape: ";
//         for (auto d : s) std::cout << d << " ";
//         std::cout << ")";
//     }

//     std::cout << std::endl;
// }

// #include "tensor/dtensor.h"
// #include <cuda_runtime.h>
// #include <nccl.h>
// #include <fstream>
// #include <iostream>
// #include <filesystem>

// CachingAllocator gAllocator;

// // =========================================================
// // Constructor / Destructor
// // =========================================================
// DTensor::DTensor(int rank, int world_size, ProcessGroup* pg)
//     : rank_(rank),
//       world_size_(world_size),
//       size_(0),
//       pg_(pg),
//       stream_(nullptr),
//       data_block_(nullptr),
//       temp_block_(nullptr),
//       shape_({1}),
//       tensor_(OwnTensor::Shape{{1}},
//               OwnTensor::TensorOptions()
//                   .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
//                   .with_dtype(OwnTensor::Dtype::Float32)),
//       temp_tensor_(OwnTensor::Shape{{1}},
//                    OwnTensor::TensorOptions()
//                        .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
//                        .with_dtype(OwnTensor::Dtype::Float32)) {
//     cudaSetDevice(rank_);
//     stream_ = pg_->getStream();
// }

// DTensor::~DTensor() {
//     cudaStreamSynchronize(stream_);
//     if (data_block_) gAllocator.freeMemory(data_block_);
//     if (temp_block_) gAllocator.freeMemory(temp_block_);
// }

// // =========================================================
// // Data Setup
// // =========================================================
// void DTensor::setData(const std::vector<float>& host_data, const std::vector<int>& shape) {
//     size_ = static_cast<int>(host_data.size());
//     shape_ = shape.empty() ? std::vector<int>{size_} : shape;
//     dtype_ = "float32";

//     OwnTensor::Shape shape_obj;
//     shape_obj.dims.assign(shape_.begin(), shape_.end());

//     OwnTensor::TensorOptions opts;
//     opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
//                .with_dtype(OwnTensor::Dtype::Float32);

//     tensor_ = OwnTensor::Tensor(shape_obj, opts);
//     temp_tensor_ = OwnTensor::Tensor(shape_obj, opts);
//     tensor_.set_data(host_data);

//     if (data_block_) gAllocator.freeMemory(data_block_);
//     if (temp_block_) gAllocator.freeMemory(temp_block_);
//     data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
//     temp_block_ = gAllocator.allocateMemory(size_ * world_size_ * sizeof(float), stream_);
// }

// std::vector<float> DTensor::getData() const {
//     std::vector<float> host_data(size_);
//     cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
//                     size_ * sizeof(float),
//                     cudaMemcpyDeviceToHost, stream_);
//     cudaStreamSynchronize(stream_);
//     return host_data;
// }

// // =========================================================
// // Collectives
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
// // TensorOps (Bridge Integration)
// // =========================================================
// #define DEFINE_TENSOR_OP(func, op_name) \
// DTensor DTensor::func(const DTensor& other) const { \
//     OwnTensor::Tensor result = TensorOpsBridge::op_name(tensor_, other.tensor_); \
//     DTensor out(rank_, world_size_, pg_); \
//     out.tensor_ = result; \
//     out.dtype_ = "float32"; \
//     out.size_ = 1; \
//     out.shape_.clear(); \
//     for (auto d : result.shape().dims) { \
//         out.size_ *= d; \
//         out.shape_.push_back(static_cast<int>(d)); \
//     } \
//     return out; \
// }

// DEFINE_TENSOR_OP(add, add)
// DEFINE_TENSOR_OP(sub, sub)
// DEFINE_TENSOR_OP(mul, mul)
// DEFINE_TENSOR_OP(div, div)
// DEFINE_TENSOR_OP(matmul, matmul)

// // =========================================================
// // Reshape
// // =========================================================
// DTensor DTensor::reshape(int rows, int cols) const {
//     return reshape({rows, cols});
// }

// DTensor DTensor::reshape(const std::vector<int>& new_shape) const {
//     int new_size = 1;
//     for (int d : new_shape) new_size *= d;
//     if (new_size != size_)
//         throw std::runtime_error("DTensor::reshape: element count mismatch");

//     OwnTensor::Shape shape_obj;
//     shape_obj.dims.assign(new_shape.begin(), new_shape.end());
//     OwnTensor::Tensor reshaped_tensor = tensor_.reshape(shape_obj);

//     DTensor out(rank_, world_size_, pg_);
//     out.tensor_ = reshaped_tensor;
//     out.shape_ = new_shape;
//     out.size_ = size_;
//     out.dtype_ = dtype_;
//     return out;
// }

// // =========================================================
// // Checkpointing (N-D Safe)
// // =========================================================
// void DTensor::saveCheckpoint(const std::string& path) const {
//     std::vector<float> host_data(size_);
//     cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
//                     size_ * sizeof(float),
//                     cudaMemcpyDeviceToHost, stream_);
//     cudaStreamSynchronize(stream_);

//     std::ofstream file(path, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for writing: " << path << std::endl;
//         return;
//     }

//     int ndim = static_cast<int>(shape_.size());
//     file.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
//     file.write(reinterpret_cast<const char*>(shape_.data()), ndim * sizeof(int));
//     file.write(dtype_.c_str(), dtype_.size() + 1);
//     file.write(reinterpret_cast<const char*>(host_data.data()), size_ * sizeof(float));
//     file.close();

//     std::cout << "[Rank " << rank_ << "] Checkpoint saved: " << path << " (" << ndim << "D, " << size_ << " elements)\n";
// }

// void DTensor::loadCheckpoint(const std::string& path) {
//     std::ifstream file(path, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint: " << path << std::endl;
//         return;
//     }

//     int ndim = 0;
//     file.read(reinterpret_cast<char*>(&ndim), sizeof(int));

//     // Backward compatibility for legacy checkpoints
//     if (ndim <= 0 || ndim > 8) {
//         file.seekg(0);
//         int saved_size, saved_shape;
//         char dtype_buf[32];
//         file.read(reinterpret_cast<char*>(&saved_size), sizeof(int));
//         file.read(reinterpret_cast<char*>(&saved_shape), sizeof(int));
//         file.read(dtype_buf, sizeof(dtype_buf));
//         dtype_ = std::string(dtype_buf);
//         shape_ = {saved_shape};
//         ndim = 1;
//     } else {
//         shape_.resize(ndim);
//         file.read(reinterpret_cast<char*>(shape_.data()), ndim * sizeof(int));
//         char dtype_buf[32];
//         file.read(dtype_buf, sizeof(dtype_buf));
//         dtype_ = std::string(dtype_buf);
//     }

//     size_ = 1;
//     for (int d : shape_) size_ *= d;

//     std::vector<float> host_data(size_);
//     file.read(reinterpret_cast<char*>(host_data.data()), size_ * sizeof(float));
//     file.close();

//     OwnTensor::Shape shape_obj;
//     shape_obj.dims.assign(shape_.begin(), shape_.end());
//     OwnTensor::TensorOptions opts;
//     opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
//                .with_dtype(OwnTensor::Dtype::Float32);

//     tensor_ = OwnTensor::Tensor(shape_obj, opts);
//     tensor_.set_data(host_data);

//     std::cout << "[Rank " << rank_ << "] Checkpoint loaded: " << path
//               << " (" << ndim << "D, " << size_ << " elements)\n";
// }

// // =========================================================
// // Recursive Pretty Printer
// // =========================================================
// void DTensor::printRecursive(const std::vector<float>& data,
//                              const std::vector<int>& dims,
//                              int dim,
//                              int offset) const {
//     if (dim == dims.size() - 1) {
//         std::cout << "[";
//         for (int i = 0; i < dims[dim]; ++i) {
//             if (i > 0) std::cout << ", ";
//             std::cout << data[offset + i];
//         }
//         std::cout << "]";
//         return;
//     }

//     std::cout << "[";
//     int stride = 1;
//     for (int i = dim + 1; i < dims.size(); ++i) stride *= dims[i];
//     for (int i = 0; i < dims[dim]; ++i) {
//         if (i > 0) std::cout << ", ";
//         printRecursive(data, dims, dim + 1, offset + i * stride);
//     }
//     std::cout << "]";
// }

// void DTensor::print() const {
//     if (size_ <= 0 || tensor_.data<float>() == nullptr) {
//         std::cerr << "[Rank " << rank_ << "] Tensor is empty or uninitialized!\n";
//         return;
//     }

//     std::vector<float> host_data(size_);
//     cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
//                     size_ * sizeof(float),
//                     cudaMemcpyDeviceToHost,
//                     stream_);
//     cudaStreamSynchronize(stream_);

//     std::cout << "[Rank " << rank_ << "] ";
//     printRecursive(host_data, shape_, 0, 0);
//     std::cout << "\n";
// }

#include "tensor/dtensor.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <fstream>
#include <iostream>
#include <filesystem>

CachingAllocator gAllocator;

// =========================================================
// Constructor / Destructor
// =========================================================
DTensor::DTensor(int rank, int world_size, ProcessGroup* pg)
    : rank_(rank),
      world_size_(world_size),
      size_(0),
      pg_(pg),
      stream_(nullptr),
      data_block_(nullptr),
      temp_block_(nullptr),
      shape_({1}),
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

__attribute__((used)) DTensor::~DTensor() {
    cudaStreamSynchronize(stream_);
    if (data_block_) gAllocator.freeMemory(data_block_);
    if (temp_block_) gAllocator.freeMemory(temp_block_);
}

// =========================================================
// Data Setup
// =========================================================
void DTensor::setData(const std::vector<float>& host_data, const std::vector<int>& shape) {
    size_ = static_cast<int>(host_data.size());
    shape_ = shape.empty() ? std::vector<int>{size_} : shape;
    dtype_ = "float32";

    OwnTensor::Shape shape_obj;
    shape_obj.dims.assign(shape_.begin(), shape_.end());

    OwnTensor::TensorOptions opts;
    opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
               .with_dtype(OwnTensor::Dtype::Float32);

    tensor_ = OwnTensor::Tensor(shape_obj, opts);
    temp_tensor_ = OwnTensor::Tensor(shape_obj, opts);
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
#define DEFINE_TENSOR_OP(func, op_name) \
DTensor DTensor::func(const DTensor& other) const { \
    OwnTensor::Tensor result = TensorOpsBridge::op_name(tensor_, other.tensor_); \
    DTensor out(rank_, world_size_, pg_); \
    out.tensor_ = result; \
    out.dtype_ = "float32"; \
    out.size_ = 1; \
    out.shape_.clear(); \
    for (auto d : result.shape().dims) { \
        out.size_ *= d; \
        out.shape_.push_back(static_cast<int>(d)); \
    } \
    return out; \
}

DEFINE_TENSOR_OP(add, add)
DEFINE_TENSOR_OP(sub, sub)
DEFINE_TENSOR_OP(mul, mul)
DEFINE_TENSOR_OP(div, div)
DEFINE_TENSOR_OP(matmul, matmul)

// =========================================================
// Reshape
// =========================================================
DTensor DTensor::reshape(int rows, int cols) const {
    return reshape({rows, cols});
}

DTensor DTensor::reshape(const std::vector<int>& new_shape) const {
    int new_size = 1;
    for (int d : new_shape) new_size *= d;
    if (new_size != size_)
        throw std::runtime_error("DTensor::reshape: element count mismatch");

    OwnTensor::Shape shape_obj;
    shape_obj.dims.assign(new_shape.begin(), new_shape.end());
    OwnTensor::Tensor reshaped_tensor = tensor_.reshape(shape_obj);

    DTensor out(rank_, world_size_, pg_);
    out.tensor_ = reshaped_tensor;
    out.shape_ = new_shape;
    out.size_ = size_;
    out.dtype_ = dtype_;
    return out;
}

// =========================================================
// Checkpointing (N-D Safe)
// =========================================================
void DTensor::saveCheckpoint(const std::string& path) const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for writing: " << path << std::endl;
        return;
    }

    int ndim = static_cast<int>(shape_.size());
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
    file.write(reinterpret_cast<const char*>(shape_.data()), ndim * sizeof(int));
    file.write(dtype_.c_str(), dtype_.size() + 1);
    file.write(reinterpret_cast<const char*>(host_data.data()), size_ * sizeof(float));
    file.close();

    std::cout << "[Rank " << rank_ << "] Checkpoint saved: " << path << " (" << ndim << "D, " << size_ << " elements)\n";
}

void DTensor::loadCheckpoint(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint: " << path << std::endl;
        return;
    }

    int ndim = 0;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(int));

    if (ndim <= 0 || ndim > 8) {
        file.seekg(0);
        int saved_size, saved_shape;
        char dtype_buf[32];
        file.read(reinterpret_cast<char*>(&saved_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&saved_shape), sizeof(int));
        file.read(dtype_buf, sizeof(dtype_buf));
        dtype_ = std::string(dtype_buf);
        shape_ = {saved_shape};
        ndim = 1;
    } else {
        shape_.resize(ndim);
        file.read(reinterpret_cast<char*>(shape_.data()), ndim * sizeof(int));
        char dtype_buf[32];
        file.read(dtype_buf, sizeof(dtype_buf));
        dtype_ = std::string(dtype_buf);
    }

    size_ = 1;
    for (int d : shape_) size_ *= d;

    std::vector<float> host_data(size_);
    file.read(reinterpret_cast<char*>(host_data.data()), size_ * sizeof(float));
    file.close();

    OwnTensor::Shape shape_obj;
    shape_obj.dims.assign(shape_.begin(), shape_.end());
    OwnTensor::TensorOptions opts;
    opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
               .with_dtype(OwnTensor::Dtype::Float32);

    tensor_ = OwnTensor::Tensor(shape_obj, opts);
    tensor_.set_data(host_data);

    std::cout << "[Rank " << rank_ << "] Checkpoint loaded: " << path
              << " (" << ndim << "D, " << size_ << " elements)\n";
}

// =========================================================
// Recursive Pretty Printer
// =========================================================
void DTensor::printRecursive(const std::vector<float>& data,
                             const std::vector<int>& dims,
                             int dim,
                             int offset) const {
    if (dim == dims.size() - 1) {
        std::cout << "[";
        for (int i = 0; i < dims[dim]; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << data[offset + i];
        }
        std::cout << "]";
        return;
    }

    std::cout << "[";
    int stride = 1;
    for (int i = dim + 1; i < dims.size(); ++i) stride *= dims[i];
    for (int i = 0; i < dims[dim]; ++i) {
        if (i > 0) std::cout << ", ";
        printRecursive(data, dims, dim + 1, offset + i * stride);
    }
    std::cout << "]";
}

void DTensor::print() const {
    if (size_ <= 0 || tensor_.data<float>() == nullptr) {
        std::cerr << "[Rank " << rank_ << "] Tensor is empty or uninitialized!\n";
        return;
    }

    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronize(stream_);

    std::cout << "[Rank " << rank_ << "] ";
    printRecursive(host_data, shape_, 0, 0);
    std::cout << "\n";
}

// // =========================================================
// // Force linker to retain DTensor symbols
// // =========================================================
// extern "C" void __force_link_dtensor_symbols() {
//     // Explicitly construct and destroy objects to force symbol emission
//     DTensor* temp = new DTensor(0, 1, nullptr);
//     delete temp;

//     {
//         DTensor local(0, 1, nullptr);
//         (void)local;
//     }
// }

