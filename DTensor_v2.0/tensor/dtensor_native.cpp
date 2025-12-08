#include "tensor/dtensor_native.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <numeric>
#include <stdexcept>
#include <sstream>

// Use TensorOpsBridge for ops (still more direct than going through DTensor wrapper)
#include "bridge/tensor_ops_bridge.h"

namespace OwnTensor {

DTensorNative::DTensorNative(std::shared_ptr<DeviceMesh> device_mesh, std::shared_ptr<ProcessGroup> pg)
    : rank_(pg->getRank()),
      world_size_(pg->getWorldSize()),
      device_mesh_(device_mesh),
      pg_(pg),
      stream_(pg->getStream()),
      layout_(Layout::replicated(device_mesh, {})),
      size_(0),
      data_block_(nullptr),
      temp_block_(nullptr),
      shape_({0}),
      tensor_(OwnTensor::Shape{{1}},
              OwnTensor::TensorOptions()
                  .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
                  .with_dtype(OwnTensor::Dtype::Float32)),
      temp_tensor_(tensor_) {
    cudaSetDevice(rank_);
}

DTensorNative::DTensorNative(std::shared_ptr<DeviceMesh> device_mesh,
                             std::shared_ptr<ProcessGroup> pg,
                             const OwnTensor::Tensor& local_tensor,
                             const Layout& layout)
    : rank_(pg->getRank()),
      world_size_(pg->getWorldSize()),
      device_mesh_(device_mesh),
      pg_(pg),
      stream_(pg->getStream()),
      layout_(layout),
      tensor_(local_tensor),
      temp_tensor_(local_tensor) {
    cudaSetDevice(rank_);

    shape_ = layout_.get_local_shape(rank_);
    size_ = 1;
    for (int d : shape_) size_ *= d;

    data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
    temp_block_ = gAllocator.allocateMemory(layout.global_numel() * sizeof(float), stream_);
}

DTensorNative::~DTensorNative() {
    cudaStreamSynchronize(stream_);
    if (data_block_) gAllocator.freeMemory(data_block_);
    if (temp_block_) gAllocator.freeMemory(temp_block_);
}

void DTensorNative::setData(const std::vector<float>& host_data, const Layout& layout) {
    layout_ = layout;

    std::vector<int> local_shape = layout_.get_local_shape(rank_);
    shape_ = local_shape;

    size_ = 1;
    for (int d : local_shape) size_ *= d;

    if (host_data.size() != (size_t)size_) {
        std::ostringstream oss;
        oss << "DTensorNative::setData: host_data size (" << host_data.size()
            << ") does not match calculated local shard size (" << size_ << ")."
            << " Rank: " << rank_ << ", " << layout_.describe(rank_);
        throw std::runtime_error(oss.str());
    }

    OwnTensor::Shape shape_obj;
    shape_obj.dims.assign(local_shape.begin(), local_shape.end());

    OwnTensor::TensorOptions opts;
    opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
               .with_dtype(OwnTensor::Dtype::Float32);

    tensor_ = OwnTensor::Tensor(shape_obj, opts);
    tensor_.set_data(host_data);

    temp_tensor_ = OwnTensor::Tensor(shape_obj, opts);

    if (data_block_) gAllocator.freeMemory(data_block_);
    if (temp_block_) gAllocator.freeMemory(temp_block_);
    data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
    temp_block_ = gAllocator.allocateMemory(layout.global_numel() * sizeof(float), stream_);
}

std::vector<float> DTensorNative::getData() const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return host_data;
}

// Matmul implementation - DIRECT OwnTensor calls (no bridge)
DTensorNative DTensorNative::matmul(const DTensorNative& other) const {
    const Layout& a_layout = this->layout_;
    const Layout& b_layout = other.get_layout();

    auto a_placement = a_layout.get_placement(0);
    auto b_placement = b_layout.get_placement(0);

    // Column-parallel: X [M, K] @ W [K, N/P] -> Y [M, N/P]
    if (a_placement->type() == PlacementType::REPLICATE &&
        b_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(b_placement.get())->dim() == 1) {
        return _column_parallel_matmul(other);
    }

    // Row-parallel: X [M, K/P] @ W [K/P, N] -> Y_partial [M, N] -> AllReduce -> Y [M, N]
    if (a_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(a_placement.get())->dim() == 1 &&
        b_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(b_placement.get())->dim() == 0) {
         return _row_parallel_matmul(other);
    }

    std::ostringstream oss;
    oss << "DTensorNative::matmul: This sharding combination is not implemented!\\n"
        << "  Layout A: " << a_layout.describe(rank_) << "\\n"
        << "  Layout B: " << b_layout.describe(rank_);
    throw std::runtime_error(oss.str());
}

DTensorNative DTensorNative::_column_parallel_matmul(const DTensorNative& other) const {
    // Direct TensorOps call (no DTensor layer)
    Tensor Y_shard = TensorOpsBridge::matmul(this->tensor_, other.local_tensor());

    std::vector<int> Y_global_shape = {
        this->layout_.get_global_shape()[0],
        other.get_layout().get_global_shape()[1]
    };
    Layout Y_layout(device_mesh_, Y_global_shape, ShardingType::SHARDED, 1);

    return DTensorNative(device_mesh_, pg_, Y_shard, Y_layout);
}

DTensorNative DTensorNative::_row_parallel_matmul(const DTensorNative& other) const {
    // Direct TensorOps call (no DTensor layer)
    Tensor Y_partial = TensorOpsBridge::matmul(this->tensor_, other.local_tensor());

    std::vector<int> Y_global_shape = {
        this->layout_.get_global_shape()[0],
        other.get_layout().get_global_shape()[1]
    };
    Layout Y_layout = Layout::replicated(device_mesh_, Y_global_shape);
    DTensorNative Y_out(device_mesh_, pg_, Y_partial, Y_layout);

    // AllReduce with SUM
    pg_->allReduce<float>(Y_out.tensor_.data<float>(), Y_out.tensor_.data<float>(), 
                          Y_out.size_, ncclFloat, ncclSum)->wait();

    return Y_out;
}

} // namespace OwnTensor
