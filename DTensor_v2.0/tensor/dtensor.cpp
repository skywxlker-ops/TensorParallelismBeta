/**
 * @file dtensor.cpp
 * @brief Distributed Tensor Implementation for Tensor Parallelism
 *
 * DTensor provides a distributed tensor abstraction that handles:
 * - Multi-GPU tensor distribution (sharded and replicated layouts)
 * - NCCL-based collective operations
 * - Tensor-parallel matrix multiplication
 * - Multi-stream execution for computation/communication overlap
 */

#include "tensor/dtensor.h"
#include "tensor/fused_shard_kernel.cuh"
#include "tensor/distributed_loss_kernels.cuh"
#include "autograd/ops_template.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <sstream>

// Global Allocator
// ============================================================================

// gAllocator is now OwnTensor::gAllocator, defined in Tensor-Implementations/src/device/cachingAllocator.cpp

// Initialize static members
cudaStream_t DTensor::shared_compute_stream_ = nullptr;
cudaStream_t DTensor::shared_comm_stream_ = nullptr;
cudaStream_t DTensor::shared_data_stream_ = nullptr;
cudaEvent_t DTensor::shared_compute_event_ = nullptr;
cudaEvent_t DTensor::shared_comm_event_ = nullptr;
bool DTensor::streams_initialized_ = false;

void DTensor::init_shared_streams() {
    if (streams_initialized_) return;
    
    cudaSetDevice(rank_);
    cudaStreamCreate(&shared_compute_stream_);
    cudaStreamCreate(&shared_comm_stream_);
    cudaStreamCreate(&shared_data_stream_);
    cudaEventCreateWithFlags(&shared_compute_event_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&shared_comm_event_, cudaEventDisableTiming);
    
    streams_initialized_ = true;
}

// ============================================================================
// Static Helper Functions
// ============================================================================

namespace {

/**
 * Convert std::vector<int64_t> to OwnTensor::Shape
 */
OwnTensor::Shape toShape(const std::vector<int64_t>& dims) {
    OwnTensor::Shape shape;
    shape.dims.assign(dims.begin(), dims.end());
    return shape;
}

/**
 * Get TensorOptions for the given rank
 */
OwnTensor::TensorOptions getOpts(int rank, Dtype dtype = Dtype::Float32) {
    return OwnTensor::TensorOptions()
        .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
        .with_dtype(dtype);
}

/**
 * Calculate total element count from shape
 */
size_t numelFromShape(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

} // anonymous namespace

// ============================================================================
// Constructors & Destructor
// ============================================================================

DTensor::DTensor(std::shared_ptr<DeviceMesh> device_mesh, std::shared_ptr<ProcessGroupNCCL> pg)
    : rank_(pg->get_rank()),
      world_size_(pg->get_worldsize()),
      device_mesh_(device_mesh),
      pg_(pg),
      layout_(Layout::replicated(*device_mesh, {})),
      size_(0),
      shape_({0}),
      tensor_(OwnTensor::Shape{{1}},
              OwnTensor::TensorOptions()
                  .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
                  .with_dtype(OwnTensor::Dtype::Float32)),
      temp_tensor_(tensor_),
      dtype_enum_(Dtype::Float32)
{
    init_shared_streams();
    compute_stream_ = shared_compute_stream_;
    comm_stream_ = shared_comm_stream_;
    data_stream_ = shared_data_stream_;
    compute_event_ = shared_compute_event_;
    comm_event_ = shared_comm_event_;
}

DTensor::DTensor(std::shared_ptr<DeviceMesh> device_mesh,
                 std::shared_ptr<ProcessGroupNCCL> pg,
                 const OwnTensor::Tensor& local_tensor,
                 const Layout& layout)
    : rank_(pg->get_rank()),
      world_size_(pg->get_worldsize()),
      device_mesh_(device_mesh),
      pg_(pg),
      layout_(layout),
      tensor_(local_tensor),
      temp_tensor_(local_tensor),
      dtype_enum_(local_tensor.dtype())
{
    shape_ = layout_.get_local_shape(rank_);
    size_ = numelFromShape(shape_);
    
    init_shared_streams();
    compute_stream_ = shared_compute_stream_;
    comm_stream_ = shared_comm_stream_;
    data_stream_ = shared_data_stream_;
    compute_event_ = shared_compute_event_;
    comm_event_ = shared_comm_event_;
    
    // Initialize requires_grad_ from the local tensor
    requires_grad_ = tensor_.requires_grad();
}

// Constructor matching friend's API: DTensor(device_mesh, pg, layout)
DTensor::DTensor(DeviceMesh& device_mesh, std::shared_ptr<ProcessGroupNCCL> pg, const Layout& layout)
    : rank_(pg->get_rank()),
      world_size_(pg->get_worldsize()),
      device_mesh_(std::make_shared<DeviceMesh>(device_mesh)),  // Copy to shared_ptr
      pg_(pg),
      layout_(layout),
      size_(0),
      shape_(layout.get_local_shape(pg->get_rank())),
      tensor_(toShape(layout.get_local_shape(pg->get_rank())),
              OwnTensor::TensorOptions()
                  .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, pg->get_rank()))
                  .with_dtype(OwnTensor::Dtype::Float32)),
      temp_tensor_(tensor_),
      dtype_enum_(Dtype::Float32)
{
    init_shared_streams();
    compute_stream_ = shared_compute_stream_;
    comm_stream_ = shared_comm_stream_;
    data_stream_ = shared_data_stream_;
    compute_event_ = shared_compute_event_;
    comm_event_ = shared_comm_event_;
    
    size_ = numelFromShape(shape_);
}

DTensor::~DTensor() {
    // Shared streams are managed statically, so no cleanup needed here.
    // This avoids RAII bugs when temporary DTensors are destroyed.
}

// ============================================================================
// Data Initialization
// ============================================================================

void DTensor::setData(const std::vector<float>& host_data, const Layout& layout) {
    layout_ = layout;
    shape_ = layout_.get_local_shape(rank_);
    size_ = numelFromShape(shape_);

    if (host_data.size() != static_cast<size_t>(size_)) {
        std::ostringstream oss;
        oss << "DTensor::setData: host_data size (" << host_data.size()
            << ") does not match local shard size (" << size_ << ")."
            << " Rank: " << rank_ << ", " << layout_.describe(rank_);
        throw std::runtime_error(oss.str());
    }

    tensor_ = OwnTensor::Tensor(toShape(shape_), getOpts(rank_, dtype_enum_));
    temp_tensor_ = tensor_;

    if (dtype_enum_ == Dtype::Float32) {
        tensor_.set_data(host_data);
    } else {
        // Robust conversion for non-float tensors (cast float → target int)
        if (dtype_enum_ == Dtype::UInt16) {
            std::vector<uint16_t> casted(host_data.size());
            for (size_t i = 0; i < host_data.size(); ++i) casted[i] = static_cast<uint16_t>(host_data[i]);
            tensor_.set_data(casted);
        } else if (dtype_enum_ == Dtype::Int32) {
            std::vector<int32_t> casted(host_data.size());
            for (size_t i = 0; i < host_data.size(); ++i) casted[i] = static_cast<int32_t>(host_data[i]);
            tensor_.set_data(casted);
        } else {
            throw std::runtime_error("DTensor::setData: unsupported target dtype for conversion");
        }
    }
}

void DTensor::setDataFromRoot(const std::vector<float>& host_data, const Layout& layout, int root) {
    layout_ = layout;
    
    std::vector<int64_t> global_shape = layout.get_global_shape();
    std::vector<int64_t> local_shape = layout.get_local_shape(rank_);
    size_t global_size = numelFromShape(global_shape);
    size_t local_size = numelFromShape(local_shape);
    
    // Validate input on root rank
    if (rank_ == root && host_data.size() != global_size) {
        std::ostringstream oss;
        oss << "DTensor::setDataFromRoot: Root rank " << root
            << " host_data size (" << host_data.size()
            << ") does not match global tensor size (" << global_size << ").";
        throw std::runtime_error(oss.str());
    }
    
    // Case 1: Replicated layout - Broadcast full tensor
    if (layout.is_replicated()) {
        tensor_ = OwnTensor::Tensor(toShape(global_shape), getOpts(rank_));
        
        if (rank_ == root) {
            tensor_.set_data(host_data);
        }
        
        pg_->broadcast(tensor_.data<float>(), tensor_.data<float>(),
                       global_size, OwnTensor::Dtype::Float32, root, true);
        
        shape_ = global_shape;
        size_ = global_size;
        return;
    }
    
    // Case 2: Sharded layout
    tensor_ = OwnTensor::Tensor(toShape(local_shape), getOpts(rank_));
    
    if (layout.get_shard_dim() == 0) {
        // Row-sharded: scatter contiguous chunks
        OwnTensor::Tensor* temp_full_ptr = nullptr;
        if (rank_ == root) {
            temp_full_ptr = new OwnTensor::Tensor(toShape(global_shape), getOpts(rank_));
            temp_full_ptr->set_data(host_data);
        }
        
        pg_->scatter(
            temp_full_ptr ? temp_full_ptr->data<float>() : nullptr,
            tensor_.data<float>(),
            local_size,
            OwnTensor::Dtype::Float32,
            root,
            true
        );
        
        delete temp_full_ptr;
        
    } else if (layout.get_shard_dim() == 1) {
        // Column-sharded: broadcast then extract
        OwnTensor::Tensor temp_full(toShape(global_shape), getOpts(rank_));
        
        if (rank_ == root) {
            temp_full.set_data(host_data);
        }
        
        pg_->broadcast(temp_full.data<float>(), temp_full.data<float>(),
                       global_size, OwnTensor::Dtype::Float32, root, true);
        _extract_local_shard(temp_full, layout);
    }
    
    shape_ = local_shape;
    size_ = local_size;
}

std::vector<float> DTensor::getData() const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float), cudaMemcpyDeviceToHost, data_stream_);
    cudaStreamSynchronize(data_stream_);
    return host_data;
}

// ============================================================================
// Collective Operations (Synchronous)
// ============================================================================

void DTensor::allReduce() {
    pg_->all_reduce(tensor_.data<float>(), tensor_.data<float>(),
                    size_, OwnTensor::Dtype::Float32, sum, true);
}


void DTensor::reduceScatter() {
    size_t count_per_shard = size_ / world_size_;
    
    pg_->reduce_scatter(tensor_.data<float>(), tensor_.data<float>(),
                        count_per_shard, OwnTensor::Dtype::Float32, sum, true);

    std::vector<int64_t> local_shape = layout_.get_local_shape(rank_);
    ensureTempTensor(local_shape);

    cudaMemcpyAsync(temp_tensor_.data<float>(),
                    tensor_.data<float>() + rank_ * count_per_shard,
                    count_per_shard * sizeof(float),
                    cudaMemcpyDeviceToDevice, data_stream_);
    cudaStreamSynchronize(data_stream_);

    std::swap(tensor_, temp_tensor_);
    shape_ = local_shape;
    size_ = count_per_shard;
}

void DTensor::allGather() {
    size_t count_per_rank = size_;
    std::vector<int64_t> global_shape = layout_.get_global_shape();
    ensureTempTensor(global_shape);

    pg_->all_gather(tensor_.data<float>(), temp_tensor_.data<float>(),
                    count_per_rank, OwnTensor::Dtype::Float32, true);

    std::swap(tensor_, temp_tensor_);
    shape_ = global_shape;
    size_ = tensor_.numel();
}

void DTensor::broadcast(int root) {
    pg_->broadcast(tensor_.data<float>(), tensor_.data<float>(),
                   size_, OwnTensor::Dtype::Float32, root, true);
}

void DTensor::sync() {
    recordComputeDone();
    waitForCompute();
    pg_->all_reduce(tensor_.data<float>(), tensor_.data<float>(),
                    size_, OwnTensor::Dtype::Float32, sum, true);
    recordCommDone();
}

// ============================================================================
// Collective Operations (Asynchronous)
// ============================================================================

void DTensor::allReduce_async() {
    recordComputeDone();
    waitForCompute();
    pg_->all_reduce_async(tensor_.data<float>(), tensor_.data<float>(),
                          size_, OwnTensor::Dtype::Float32, sum);
    recordCommDone();
}

// void DTensor::sync_async() {
//     recordComputeDone();
//     waitForCompute();
//     pg_->all_reduce_async(tensor_.data<float>(), tensor_.data<float>(),
//                           size_, OwnTensor::Dtype::Float32, sum);
//     recordCommDone();
// }

void DTensor::reduceScatter_async() {
    size_t count_per_shard = size_ / world_size_;
    
    recordComputeDone();
    waitForCompute();
    pg_->reduce_scatter_async(tensor_.data<float>(), tensor_.data<float>(),
                              count_per_shard, OwnTensor::Dtype::Float32, sum);
    
    std::vector<int64_t> local_shape = layout_.get_local_shape(rank_);
    ensureTempTensor(local_shape);
    cudaMemcpyAsync(temp_tensor_.data<float>(),
                    tensor_.data<float>() + rank_ * count_per_shard,
                    count_per_shard * sizeof(float),
                    cudaMemcpyDeviceToDevice, data_stream_);
    
    std::swap(tensor_, temp_tensor_);
    shape_ = local_shape;
    size_ = count_per_shard;
    recordCommDone();
}

void DTensor::allGather_async() {
    size_t count_per_rank = size_;
    
    recordComputeDone();
    waitForCompute();
    
    std::vector<int64_t> global_shape = layout_.get_global_shape();
    ensureTempTensor(global_shape);
    pg_->all_gather_async(tensor_.data<float>(), temp_tensor_.data<float>(),
                          count_per_rank, OwnTensor::Dtype::Float32);
    
    std::swap(tensor_, temp_tensor_);
    shape_ = global_shape;
    size_ = tensor_.numel();
    recordCommDone();
}

// ============================================================================
// Layout Transformations (In-Place)
// ============================================================================

void DTensor::replicate(int root) {
    std::vector<int64_t> global_shape = layout_.get_global_shape();
    size_t total_numel = numelFromShape(global_shape);
    
    if (tensor_.numel() != total_numel) {
        tensor_ = OwnTensor::Tensor(toShape(global_shape),
                                    OwnTensor::TensorOptions()
                                        .with_device(tensor_.device())
                                        .with_dtype(tensor_.dtype()));
    }
    
    pg_->broadcast(tensor_.data<float>(), tensor_.data<float>(),
                   total_numel, OwnTensor::Dtype::Float32, root, true);
    
    layout_ = Layout::replicated(*device_mesh_, std::vector<int64_t>(global_shape.begin(), global_shape.end()));
    shape_ = global_shape;
    size_ = total_numel;
}

void DTensor::shard(int dim, int root) {
    std::vector<int64_t> global_shape = layout_.get_global_shape();
    
    if (dim < 0 || dim >= static_cast<int>(global_shape.size())) {
        std::ostringstream oss;
        oss << "DTensor::shard: Invalid shard dimension " << dim
            << " for tensor with " << global_shape.size() << " dimensions";
        throw std::runtime_error(oss.str());
    }

    Layout sharded_layout(*device_mesh_, std::vector<int64_t>(global_shape.begin(), global_shape.end()), dim);
    std::vector<int64_t> local_shape = sharded_layout.get_local_shape(rank_);
    size_t shard_numel = numelFromShape(local_shape);
    
    pg_->scatter(
        tensor_.data<float>(),
        tensor_.data<float>() + rank_ * shard_numel,
        shard_numel,
        OwnTensor::Dtype::Float32,
        root,
        true
    );
    
    OwnTensor::Tensor shard_tensor(toShape(local_shape),
                                   OwnTensor::TensorOptions()
                                       .with_device(tensor_.device())
                                       .with_dtype(tensor_.dtype()));
    
    cudaMemcpyAsync(shard_tensor.data<float>(),
                    tensor_.data<float>() + rank_ * shard_numel,
                    shard_numel * sizeof(float),
                    cudaMemcpyDeviceToDevice, data_stream_);
    cudaStreamSynchronize(data_stream_);
    
    std::swap(tensor_, shard_tensor);
    layout_ = sharded_layout;
    shape_ = local_shape;
    size_ = shard_numel;
}

void DTensor::scale(float factor) {
    OwnTensor::Tensor scaled = Bridge::mul(tensor_, factor);
    tensor_ = std::move(scaled);
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

#define DEFINE_TENSOR_OP(func, op_name) \
DTensor DTensor::func(const DTensor& other) const { \
    if (!layout_.is_compatible(other.get_layout())) { \
        throw std::runtime_error("Incompatible layouts for operation " #op_name); \
    } \
    OwnTensor::Tensor result = Bridge::op_name(tensor_, other.tensor_); \
    return DTensor(device_mesh_, pg_, result, layout_); \
}

DEFINE_TENSOR_OP(add, add)
DEFINE_TENSOR_OP(sub, sub)
DEFINE_TENSOR_OP(mul, mul)
DEFINE_TENSOR_OP(div, div)

DTensor DTensor::relu() const {
    OwnTensor::Tensor result = Bridge::autograd::relu(tensor_);
    return DTensor(device_mesh_, pg_, result, layout_);
}

DTensor DTensor::mse_loss(const DTensor& target) const {
    if (!layout_.is_compatible(target.get_layout())) {
        throw std::runtime_error("Incompatible layouts for mse_loss");
    }
    OwnTensor::Tensor result = Bridge::autograd::mse_loss(tensor_, target.tensor_);
    // Loss is typically a scalar or reduced, but here it's on local shard.
    // However, MSE loss in OwnTensor::autograd returns a scalar (reduce_mean).
    // So the result is replicated (scalar).
    return DTensor(device_mesh_, pg_, result, Layout::replicated(*device_mesh_, {1}));
}

DTensor DTensor::gelu() const {
    OwnTensor::Tensor result = Bridge::autograd::gelu(tensor_);
    return DTensor(device_mesh_, pg_, result, layout_);
}

DTensor DTensor::softmax(int64_t dim) const {
    OwnTensor::Tensor result = Bridge::autograd::softmax(tensor_, dim);
    return DTensor(device_mesh_, pg_, result, layout_);
}
DTensor DTensor::cross_entropy_loss(const DTensor& target) const {
    OwnTensor::Tensor result = Bridge::autograd::categorical_cross_entropy(tensor_, target.tensor_);
    return DTensor(device_mesh_, pg_, result, Layout::replicated(*device_mesh_, {1}));
}

DTensor DTensor::sparse_cross_entropy_loss(const DTensor& target) const {
    // logits [B*T, Vocab], target [B*T] (indices)
    OwnTensor::Tensor result = Bridge::autograd::sparse_cross_entropy_loss(tensor_, target.tensor_);
    return DTensor(device_mesh_, pg_, result, Layout::replicated(*device_mesh_, {1}));
}

DTensor DTensor::embedding(const OwnTensor::Tensor& indices, DTensor& weight, int padding_idx) {
    // The embedding weight should be replicated across all ranks for simplicity
    // Indices are the same on all ranks, embedding lookup is done locally
    
    // Call Bridge autograd embedding on local weight tensor
    OwnTensor::Tensor local_result = Bridge::autograd::embedding(
        indices, weight.tensor_, padding_idx
    );
    
    // Output is [num_tokens, embedding_dim], replicated
    int64_t num_tokens = indices.numel();
    int64_t embed_dim = weight.get_layout().get_global_shape()[1];
    Layout out_layout = Layout::replicated(weight.get_layout().get_mesh(), {num_tokens, embed_dim});
    
    return DTensor(weight.device_mesh_, weight.pg_, local_result, out_layout);
}

#undef DEFINE_TENSOR_OP

// ============================================================================
// Matrix Multiplication (Tensor Parallelism)
// ============================================================================

DTensor DTensor::matmul(const DTensor& other) const {
    const Layout& a_layout = this->layout_;
    const Layout& b_layout = other.get_layout();

    auto a_placement = a_layout.get_placement(0);
    auto b_placement = b_layout.get_placement(0);

    // Column-Parallel: X [..., M, K] @ W [K, N/P] -> Y [..., M, N/P]
    // Condition: X is replicated, W is sharded on last dimension (dim 1 since W is 2D)
    if (a_placement->type() == PlacementType::REPLICATE &&
        b_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(b_placement.get())->dim() == (int)other.get_layout().get_global_shape().size() - 1) {
        return _column_parallel_matmul(other);
    }

    // Row-Parallel: X [..., M, K/P] @ W [K/P, N] -> Y_partial [..., M, N] -> AllReduce
    // Condition: X is sharded on its last dimension, W is sharded on its first dimension (dim 0)
    if (a_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(a_placement.get())->dim() == (int)this->layout_.get_global_shape().size() - 1 &&
        b_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(b_placement.get())->dim() == 0) {
        return _row_parallel_matmul(other);
    }

    // Replicated × Replicated: X [M, K] @ W [K, N] -> Y [M, N] (local matmul)
    if (a_placement->type() == PlacementType::REPLICATE &&
        b_placement->type() == PlacementType::REPLICATE) {
        OwnTensor::Tensor Y_local = Bridge::autograd::matmul(tensor_, other.tensor_);
        
        // Correct shape inference for ND: [..., M, K] @ [K, N] -> [..., M, N]
        std::vector<int64_t> Y_global_shape = a_layout.get_global_shape();
        Y_global_shape.back() = b_layout.get_global_shape().back();
        
        Layout Y_layout = Layout::replicated(*device_mesh_, Y_global_shape);
        return DTensor(device_mesh_, pg_, Y_local, Y_layout);
    }

    std::ostringstream oss;
    oss << "DTensor::matmul: Unsupported sharding combination!\n"
        << "  Layout A: " << a_layout.describe(rank_) << "\n"
        << "  Layout B: " << b_layout.describe(rank_);
    throw std::runtime_error(oss.str());
}

DTensor DTensor::_column_parallel_matmul(const DTensor& other) const {
    // Column-Parallel: X [M, K] @ W1 [K, N/P] -> H [M, N/P]
    // Use other.tensor_ directly (not local_tensor() which makes a copy)
    // so gradients accumulate to the actual weight tensor
    OwnTensor::Tensor Y_shard = Bridge::autograd::matmul(this->tensor_, other.tensor_);

    // Correct shape inference for ND: [..., M, K] @ [K, N] -> [..., M, N]
    std::vector<int64_t> Y_global_shape = this->layout_.get_global_shape();
    Y_global_shape.back() = other.get_layout().get_global_shape().back();
    
    // Shard on the last dimension (columns/vocab)
    Layout Y_layout(*device_mesh_, Y_global_shape, (int)Y_global_shape.size() - 1);
    
    return DTensor(device_mesh_, pg_, Y_shard, Y_layout);
}

DTensor DTensor::_row_parallel_matmul(const DTensor& other) const {
    // Row-Parallel: H [M, N/P] @ W2 [N/P, K] -> Y_partial [M, K]
    OwnTensor::Tensor Y_partial = Bridge::autograd::matmul(this->tensor_, other.local_tensor());

    // Correct shape inference for ND: [..., M, K] @ [K, N] -> [..., M, N]
    std::vector<int64_t> Y_global_shape = this->layout_.get_global_shape();
    Y_global_shape.back() = other.get_layout().get_global_shape().back();
    
    Layout Y_layout = Layout::replicated(*device_mesh_, Y_global_shape);
    DTensor Y_out(device_mesh_, pg_, Y_partial, Y_layout);

    Y_out.sync();
    return Y_out;
}

DTensor DTensor::reshape(const std::vector<int64_t>& new_global_shape) const {
    if (layout_.is_sharded()) {
        throw std::runtime_error("DTensor::reshape: Reshaping sharded tensors not yet implemented");
    }

    int new_local_size = numelFromShape(new_global_shape);
    if (new_local_size != size_) {
        throw std::runtime_error("DTensor::reshape: Element count mismatch");
    }

    OwnTensor::Tensor reshaped_tensor = tensor_.reshape(toShape(new_global_shape));
    Layout new_layout = Layout::replicated(*device_mesh_, std::vector<int64_t>(new_global_shape.begin(), new_global_shape.end()));

    return DTensor(device_mesh_, pg_, reshaped_tensor, new_layout);
}

// ============================================================================
// Static Factory Functions
// ============================================================================

DTensor DTensor::empty(const std::vector<int64_t>& global_shape,
                       std::shared_ptr<DeviceMesh> mesh,
                       std::shared_ptr<ProcessGroupNCCL> pg,
                       const Layout& layout,
                       Dtype dtype) {
    int rank = pg->get_rank();
    std::vector<int64_t> local_shape = layout.get_local_shape(rank);
    OwnTensor::Tensor local_tensor(toShape(local_shape), getOpts(rank, dtype));
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::zeros(const std::vector<int64_t>& global_shape,
                       std::shared_ptr<DeviceMesh> mesh,
                       std::shared_ptr<ProcessGroupNCCL> pg,
                       const Layout& layout,
                       Dtype dtype) {
    int rank = pg->get_rank();
    std::vector<int64_t> local_shape = layout.get_local_shape(rank);
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::zeros(toShape(local_shape), getOpts(rank, dtype));
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::ones(const std::vector<int64_t>& global_shape,
                      std::shared_ptr<DeviceMesh> mesh,
                      std::shared_ptr<ProcessGroupNCCL> pg,
                      const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int64_t> local_shape = layout.get_local_shape(rank);
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::ones(toShape(local_shape), getOpts(rank));
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::full(const std::vector<int64_t>& global_shape,
                      float value,
                      std::shared_ptr<DeviceMesh> mesh,
                      std::shared_ptr<ProcessGroupNCCL> pg,
                      const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int64_t> local_shape = layout.get_local_shape(rank);
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::full(toShape(local_shape), getOpts(rank), value);
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::rand(const std::vector<int64_t>& global_shape,
                      std::shared_ptr<DeviceMesh> mesh,
                      std::shared_ptr<ProcessGroupNCCL> pg,
                      const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int64_t> local_shape = layout.get_local_shape(rank);
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::rand<float>(toShape(local_shape), getOpts(rank));
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::randn(const std::vector<int64_t>& global_shape,
                       std::shared_ptr<DeviceMesh> mesh,
                       std::shared_ptr<ProcessGroupNCCL> pg,
                       const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int64_t> local_shape = layout.get_local_shape(rank);
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::randn<float>(toShape(local_shape), getOpts(rank));
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::randint(int64_t low, int64_t high,
                         const std::vector<int64_t>& global_shape,
                         std::shared_ptr<DeviceMesh> mesh,
                         std::shared_ptr<ProcessGroupNCCL> pg,
                         const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int64_t> local_shape = layout.get_local_shape(rank);
    
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::rand<float>(toShape(local_shape), getOpts(rank));
    size_t numel = numelFromShape(local_shape);
    
    std::vector<float> data(numel);
    cudaMemcpy(data.data(), local_tensor.data<float>(), numel * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < numel; ++i) {
        data[i] = static_cast<float>(static_cast<int64_t>(data[i] * (high - low) + low));
    }
    
    cudaMemcpy(local_tensor.data<float>(), data.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
    
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::redistribute(const Layout& target_layout) const {
    // Create new DTensor with same data/layout initially
    DTensor result(device_mesh_, pg_, tensor_, layout_);
    
    // Logic for redistribution
    if (layout_.is_replicated() && target_layout.is_sharded()) {
        result.shard(target_layout.get_shard_dim(), 0); 
    } else if (layout_.is_sharded() && target_layout.is_replicated()) {
        result.allGather();
    } else if (layout_.is_sharded() && target_layout.is_sharded()) {
        if (layout_.get_shard_dim() != target_layout.get_shard_dim()) {
             // Reshard: AllGather -> Shard
             result.allGather();
             result.shard(target_layout.get_shard_dim(), 0);
        }
    }
    // Note: PARTIAL layout is deprecated, so related transitions are removed.
    
    // Update internal layout to match target (in case shard/allGather didn't set it exactly as target expects, e.g. names/metadata)
    // Actually shard/allGather update layout_. 
    // But we should verify. 
    // result.layout_ is updated.
    
    return result;
}

DTensor DTensor::from_local(const OwnTensor::Tensor& local_tensor,
                            std::shared_ptr<DeviceMesh> mesh,
                            std::shared_ptr<ProcessGroupNCCL> pg,
                            const Layout& layout) {
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::distribute_tensor(const OwnTensor::Tensor& global_tensor,
                                   std::shared_ptr<DeviceMesh> mesh,
                                   std::shared_ptr<ProcessGroupNCCL> pg,
                                   const Layout& layout,
                                   int root) {
    int rank = pg->get_rank();
    
    std::vector<int64_t> global_shape = layout.get_global_shape();
    std::vector<int64_t> local_shape = layout.get_local_shape(rank);
    size_t global_size = numelFromShape(global_shape);
    size_t local_size = numelFromShape(local_shape);
    
    OwnTensor::Tensor local_tensor(toShape(local_shape), getOpts(rank));
    
    // Case 1: Replicated layout
    if (layout.is_replicated()) {
        OwnTensor::Tensor full_tensor(toShape(global_shape), getOpts(rank));
        
        if (rank == root) {
            cudaMemcpy(full_tensor.data<float>(), global_tensor.data<float>(),
                       global_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        
        pg->broadcast(full_tensor.data<float>(), full_tensor.data<float>(),
                      global_size, OwnTensor::Dtype::Float32, root, true);
        
        return DTensor(mesh, pg, full_tensor, layout);
    }
    
    // Case 2: Sharded layout
    int shard_dim = layout.get_shard_dim();
    
    if (shard_dim == 0) {
        // Row-sharded: scatter contiguous chunks
        pg->scatter(
            (rank == root) ? global_tensor.data<float>() : nullptr,
            local_tensor.data<float>(),
            local_size,
            OwnTensor::Dtype::Float32,
            root,
            true
        );
    } else if (shard_dim == 1) {
        // Column-sharded: broadcast then extract
        OwnTensor::Tensor full_tensor(toShape(global_shape), getOpts(rank));
        
        if (rank == root) {
            cudaMemcpy(full_tensor.data<float>(), global_tensor.data<float>(),
                       global_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        
        pg->broadcast(full_tensor.data<float>(), full_tensor.data<float>(),
                      global_size, OwnTensor::Dtype::Float32, root, true);
        
        // Extract column shard
        int rows = global_shape[0];
        int global_cols = global_shape[1];
        int local_cols = local_shape[1];
        int col_offset = rank * local_cols;
        
        for (int row = 0; row < rows; ++row) {
            cudaMemcpy(
                local_tensor.data<float>() + row * local_cols,
                full_tensor.data<float>() + row * global_cols + col_offset,
                local_cols * sizeof(float),
                cudaMemcpyDeviceToDevice
            );
        }
    }
    
    return DTensor(mesh, pg, local_tensor, layout);
}

// ============================================================================
// Checkpointing
// ============================================================================

void DTensor::saveCheckpoint(const std::string& path) const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float), cudaMemcpyDeviceToHost, data_stream_);
    cudaStreamSynchronize(data_stream_);

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint: " << path << std::endl;
        return;
    }

    int ndim = static_cast<int>(shape_.size());
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
    file.write(reinterpret_cast<const char*>(shape_.data()), ndim * sizeof(int64_t));
    // TODO: Update checkpoint to store Dtype enum
    std::string dtype_str = "float32"; 
    file.write(dtype_str.c_str(), dtype_str.size() + 1);
    file.write(reinterpret_cast<const char*>(host_data.data()), size_ * sizeof(float));
    file.close();

    std::cout << "[Rank " << rank_ << "] Checkpoint saved: " << path
              << " (" << ndim << "D, " << size_ << " elements)\n";
}

void DTensor::loadCheckpoint(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint: " << path << std::endl;
        return;
    }

    int ndim = 0;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(int));
    
    std::vector<int64_t> loaded_shape(ndim);
    file.read(reinterpret_cast<char*>(loaded_shape.data()), ndim * sizeof(int64_t));
    
    char dtype_buf[32];
    file.read(dtype_buf, sizeof(dtype_buf));
    // TODO: Update checkpoint to load Dtype enum
    // dtype_enum_ = ...
    
    int loaded_size = numelFromShape(loaded_shape);
    std::vector<float> host_data(loaded_size);
    file.read(reinterpret_cast<char*>(host_data.data()), loaded_size * sizeof(float));
    file.close();

    Layout loaded_layout = Layout::replicated(*device_mesh_, std::vector<int64_t>(loaded_shape.begin(), loaded_shape.end()));
    setData(host_data, loaded_layout);

    std::cout << "[Rank " << rank_ << "] Checkpoint loaded: " << path
              << " (" << ndim << "D, " << size_ << " elements)\n";
}

// ============================================================================
// Debug & Print
// ============================================================================

void DTensor::print() const {
    if (size_ <= 0 || tensor_.data<float>() == nullptr) {
        std::cerr << "[Rank " << rank_ << "] Tensor is empty or uninitialized!\n";
        return;
    }

    std::cout << layout_.describe(rank_) << std::endl;

    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float), cudaMemcpyDeviceToHost, data_stream_);
    cudaStreamSynchronize(data_stream_);

    std::cout << "[Rank " << rank_ << " Data] ";
    printRecursive(host_data, shape_, 0, 0);
    std::cout << "\n";
}

void DTensor::printRecursive(const std::vector<float>& data,
                             const std::vector<int64_t>& dims,
                             int dim, int offset) const {
    if (dims.empty() || dim < 0) return;

    if (static_cast<size_t>(dim) == dims.size() - 1) {
        std::cout << "[";
        for (int i = 0; i < dims[dim]; ++i) {
            if (i > 0) std::cout << ", ";
            if (static_cast<size_t>(offset + i) < data.size()) {
                std::cout << data[offset + i];
            }
        }
        std::cout << "]";
        return;
    }

    std::cout << "[";
    int stride = 1;
    for (size_t i = dim + 1; i < dims.size(); ++i) stride *= dims[i];
    for (int i = 0; i < dims[dim]; ++i) {
        if (i > 0) std::cout << ", ";
        printRecursive(data, dims, dim + 1, offset + i * stride);
    }
    std::cout << "]";
}

// =============================================================================
// Display & Initialization Helpers
// =============================================================================

void DTensor::display() const {
    tensor_.to_cpu().display();
}

void DTensor::rand() {
    // Generate random tensor on device
    tensor_ = OwnTensor::Tensor::rand<float>(tensor_.shape(), 
        OwnTensor::TensorOptions()
            .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
            .with_dtype(OwnTensor::Dtype::Float32));
}

void DTensor::shard_fused_transpose(int shard_dim, int root, const DTensor& source) {
    // Get source tensor dimensions
    const auto& src_shape = source.shape_;
    if (src_shape.size() != 3) {
        throw std::runtime_error("shard_fused_transpose: Source must be 3D tensor");
    }
    
    int64_t D0 = src_shape[0];
    int64_t D1 = src_shape[1]; 
    int64_t D2 = src_shape[2];
    int64_t total_src_elements = D0 * D1 * D2;
    
    // Step 1: Broadcast the full tensor from root to all ranks
    // This is the actual NCCL communication that takes time
    pg_->broadcast(
        const_cast<float*>(source.tensor_.data<float>()),
        const_cast<float*>(source.tensor_.data<float>()),
        total_src_elements,
        OwnTensor::Dtype::Float32,
        root,
        true  // blocking
    );
    
    // Step 2: Now each rank has the full tensor, extract local shard using fused kernel
    int64_t local_dim_size = (shard_dim == 1) ? (D1 / world_size_) : (D2 / world_size_);
    int64_t total_elements = D0 * local_dim_size * ((shard_dim == 1) ? D2 : D1);
    
    // Use fused shard kernel to extract local portion
    if (shard_dim == 1) {
        launch_shard_dim1_kernel(
            const_cast<float*>(source.tensor_.data<float>()),
            tensor_.data<float>(),
            D0, D1, D2,
            local_dim_size,
            rank_,
            total_elements,
            data_stream_
        );
    } else if (shard_dim == 2) {
        launch_shard_dim2_kernel(
            const_cast<float*>(source.tensor_.data<float>()),
            tensor_.data<float>(),
            D0, D1, D2,
            local_dim_size,
            rank_,
            total_elements,
            data_stream_
        );
    }
    
    cudaStreamSynchronize(data_stream_);
}

void launch_reverse_kernel(float* d_src, float* d_dst, int nx, int ny, int nz, int dim, cudaStream_t stream);

void DTensor::rotate3D(int dim, bool direction){
    
    OwnTensor::TensorOptions opts;
    opts.dtype = OwnTensor::Dtype::Float32;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_);
    OwnTensor::Tensor rtensor_(toShape(shape_), opts);
    // std::cout<<"\n Rotate3D "<<rank_<<"\n DTensor shape "<<rank_<<" ";
    // std::cout<<"["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] \n";
    // std::cout<<"\n Tensor shape "<<rank_<<" ";
    // std::cout<<"["<<tensor_.shape().dims[0]<<", "<<tensor_.shape().dims[1]<<", "<<tensor_.shape().dims[2]<<"] \n\n";

    int64_t nx = rtensor_.shape().dims[0], ny = rtensor_.shape().dims[1], nz = rtensor_.shape().dims[2];

    int64_t total_elements = (int64_t)nx * ny * nz;
 
    float* d_src = static_cast<float*>(rtensor_.data());
    float* d_dst;

    cudaMalloc(&d_dst, total_elements * sizeof(float));

    if (dim == 0)      {
        tensor_ = tensor_.transpose(1, 2);
        rtensor_ = tensor_.contiguous();
        direction ? (launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 1, data_stream_)):(launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 2, data_stream_));
        if(layout_.is_sharded()) { if (layout_.get_shard_dim() == 1 ) { layout_.set_shard_dim(2); } else if (layout_.get_shard_dim() == 2 ) {  layout_.set_shard_dim(1); } }
    }
    else if (dim == 1) {
        tensor_ = tensor_.transpose(0, 2);
        rtensor_ = tensor_.contiguous();
        direction ? (launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 0, data_stream_)):(launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 2, data_stream_));
        if(layout_.is_sharded()) { if (layout_.get_shard_dim() == 0 ) { layout_.set_shard_dim(2); } else if (layout_.get_shard_dim() == 2 ) {  layout_.set_shard_dim(0); } }
    }
    
    else {
        tensor_ = tensor_.transpose(0, 1);
        rtensor_ = tensor_.contiguous();
        direction ? (launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 0, data_stream_)):(launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 1, data_stream_));
        if(layout_.is_sharded()) { if (layout_.get_shard_dim() == 0 ) { layout_.set_shard_dim(1); } else if (layout_.get_shard_dim() == 1 ) {  layout_.set_shard_dim(0); } }
    }
    cudaMemcpyAsync(d_src, d_dst, total_elements * sizeof(float), cudaMemcpyDeviceToDevice, data_stream_);
    cudaStreamSynchronize(data_stream_);
    cudaFree(d_dst);

    // std::cout<<"\n DTensor shape "<<rank_<<"\n";
    // std::cout<<"["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] \n";
    // std::cout<<"\n Tensor shape "<<rank_<<"\n";
    // std::cout<<"["<<tensor_.shape().dims[0]<<", "<<tensor_.shape().dims[1]<<", "<<tensor_.shape().dims[2]<<"] \n";
    tensor_ = rtensor_;
    shape_ = std::vector<int64_t>(tensor_.shape().dims.begin(), tensor_.shape().dims.end());
    size_ = numelFromShape(shape_);
    // std::cout<<"\n DTensor shape "<<rank_<<"\n";
    // std::cout<<"["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] \n";
    // std::cout<<"\n Tensor shape "<<rank_<<"\n";
    // std::cout<<"["<<tensor_.shape().dims[0]<<", "<<tensor_.shape().dims[1]<<", "<<tensor_.shape().dims[2]<<"] \n";
    layout_.set_global_shape(shape_);

    
    // std::cout<<"\n Rotate3D post "<<rank_<<"\n DTensor shape "<<rank_<<" ";
    // std::cout<<"["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] \n";
    // std::cout<<"\n Tensor shape "<<rank_<<" ";
    // std::cout<<"["<<tensor_.shape().dims[0]<<", "<<tensor_.shape().dims[1]<<", "<<tensor_.shape().dims[2]<<"] \n\n";
    rtensor_.reset();

    rtensor_.release();
    
}
// ============================================================================
// Private Helper Functions
// ============================================================================

void DTensor::_extract_local_shard(const OwnTensor::Tensor& full_tensor, const Layout& layout) {
    std::vector<int64_t> local_shape = layout.get_local_shape(rank_);
    std::vector<int64_t> global_shape = layout.get_global_shape();
    
    tensor_ = OwnTensor::Tensor(toShape(local_shape),
                                OwnTensor::TensorOptions()
                                    .with_device(full_tensor.device())
                                    .with_dtype(full_tensor.dtype()));
    
    size_t local_size = tensor_.numel();
    
    if (layout.get_shard_dim() == 0) {
        // Row sharding: contiguous copy
        size_t offset = 0;
        for (int r = 0; r < rank_; ++r) {
            std::vector<int64_t> rank_shape = layout.get_local_shape(r);
            offset += numelFromShape(rank_shape);
        }
        
        cudaMemcpyAsync(tensor_.data<float>(),
                        full_tensor.data<float>() + offset,
                        local_size * sizeof(float),
                        cudaMemcpyDeviceToDevice, data_stream_);
        
    } else if (layout.get_shard_dim() == 1) {
        // Column sharding: non-contiguous 2D slice
        int rows = global_shape[0];
        int global_cols = global_shape[1];
        int local_cols = local_shape[1];
        int col_offset = rank_ * local_cols;
        
        for (int row = 0; row < rows; ++row) {
            cudaMemcpyAsync(
                tensor_.data<float>() + row * local_cols,
                full_tensor.data<float>() + row * global_cols + col_offset,
                local_cols * sizeof(float),
                cudaMemcpyDeviceToDevice,
                data_stream_
            );
        }
    }
    
    cudaStreamSynchronize(data_stream_);
}

void DTensor::ensureTempTensor(const std::vector<int64_t>& shape) {
    size_t required_size = numelFromShape(shape);
    
    if (temp_tensor_.numel() != required_size) {
        temp_tensor_ = OwnTensor::Tensor(
            toShape(shape),
            OwnTensor::TensorOptions()
                .with_device(tensor_.device())
                .with_dtype(tensor_.dtype())
        );
    }
}

void DTensor::recordComputeDone() {
    cudaEventRecord(compute_event_, compute_stream_);
}

void DTensor::recordCommDone() {
    cudaEventRecord(comm_event_, comm_stream_);
}

void DTensor::waitForCompute() {
    cudaStreamWaitEvent(comm_stream_, compute_event_, 0);
}

void DTensor::waitForComm() {
    cudaStreamWaitEvent(compute_stream_, comm_event_, 0);
}

DTensor DTensor::transpose(int dim1, int dim2) const {
    std::vector<int64_t> new_global_shape = shape_;
    std::swap(new_global_shape[dim1], new_global_shape[dim2]);
    
    // Transpose the local tensor
    OwnTensor::Tensor local_T = tensor_.transpose(dim1, dim2);
    
    // Update layout
    Layout new_layout = layout_;
    new_layout.set_global_shape(new_global_shape);
    if (layout_.is_sharded()) {
        int old_shard_dim = layout_.get_shard_dim();
        if (old_shard_dim == dim1) new_layout.set_shard_dim(dim2);
        else if (old_shard_dim == dim2) new_layout.set_shard_dim(dim1);
    }
    
    return DTensor::from_local(local_T, device_mesh_, pg_, new_layout);
}

// ============================================================================
// Autograd Interface
// ============================================================================

void DTensor::set_requires_grad(bool requires) {
    requires_grad_ = requires;
    tensor_.set_requires_grad(requires);
}

OwnTensor::Tensor DTensor::grad() const {
    OwnTensor::Tensor g = tensor_.grad_view();
    /*
    if (g.is_valid() && g.numel() > 0) {
       // Check if all zero
    }
    */
    return g;
}

void DTensor::backward(const DTensor* grad_output) {
    if (!requires_grad_) {
        throw std::runtime_error("backward() called on DTensor that doesn't require grad");
    }
    
    // std::cout << "[DEBUG] Rank " << rank_ << " DTensor::backward() called on tensor of shape " << tensor_.shape().toString() << " requires_grad=" << tensor_.requires_grad() << std::endl;
    
    if (grad_output) {
        Bridge::autograd::backward(tensor_, &grad_output->local_tensor());
    } else {
        Bridge::autograd::backward(tensor_, nullptr);
    }
}

void DTensor::zero_grad() {
    if (tensor_.requires_grad()) {
        tensor_.fill_grad<float>(0.0f);
    }
}

float DTensor::grad_norm() const {
    // Get the gradient tensor (local shard)
    OwnTensor::Tensor g = grad();
    
    // Safety check: if no gradient, return 0
    if (!g.is_valid() || g.numel() == 0) {
        return 0.0f;
    }
    
    // Compute squared values on GPU
    OwnTensor::Tensor g_sq = g * g;
    
    // Sum all elements on GPU using reduce_sum
    OwnTensor::Tensor local_sum_tensor = OwnTensor::reduce_sum(g_sq);
    
    // Copy result to CPU before accessing data pointer
    OwnTensor::Tensor local_sum_cpu = local_sum_tensor.to_cpu();
    float local_norm_sq = local_sum_cpu.data<float>()[0];
    
    // All-reduce across ranks to get global norm squared
    float global_norm_sq = 0.0f;
    pg_->all_reduce_cpu(&local_norm_sq, &global_norm_sq, 1, OwnTensor::Dtype::Float32, op_t::sum);
    
    return std::sqrt(global_norm_sq);
}

// ============================================================================
// Async Collective Operations (from _adhi_)
// ============================================================================

void DTensor::sync_async() {
    // Async sync - enqueue all-reduce but DON'T wait (deferred wait)
    // Store the work handle so we can wait later when data is actually needed
    pending_work_ = pg_->all_reduce_async(tensor_.data<float>(), tensor_.data<float>(), 
                                          size_, OwnTensor::Dtype::Float32, sum, false);
    has_pending_collective_ = true;
}

void DTensor::wait() {
    // Wait for any pending async collective to complete
    if (has_pending_collective_ && pending_work_) {
        pending_work_->wait();
        has_pending_collective_ = false;
        pending_work_ = nullptr;
    }
}

bool DTensor::has_pending_collective() const {
    return has_pending_collective_;
}

void DTensor::sync_w_autograd() {
    // Autograd-aware sync: performs all-reduce and sets up backward graph
    // Forward: Y = all_reduce_sum(X) - sum partial results across GPUs
    // Backward: automatically all-reduces gradient dY before flowing to previous ops
    
    // 1. Forward: Blocking all-reduce
    pg_->all_reduce_async(tensor_.data<float>(), tensor_.data<float>(), 
                          size_, OwnTensor::Dtype::Float32, sum, false)->wait();
    
    // 2. Autograd graph update would go here (requires AllReduceSumBackward node)
    // For now, just clear pending state
    has_pending_collective_ = false;
    pending_work_ = nullptr;
}

// ============================================================================
// Striped Attention Support (from _adhi_)
// ============================================================================

void DTensor::permute_striped(int dim) {
    if (dim < 0 || dim >= (int)shape_.size()) {
        throw std::runtime_error("DTensor::permute_striped: Invalid dimension");
    }
    
    int seq_len = shape_[dim];
    int d = world_size_;
    int n = seq_len / d;
    
    if (seq_len % d != 0) {
        std::ostringstream oss;
        oss << "DTensor::permute_striped: Sequence length (" << seq_len 
            << ") must be divisible by world_size (" << d << ")";
        throw std::runtime_error(oss.str());
    }
    
    int outer_size = 1;
    int inner_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= shape_[i];
    for (int i = dim + 1; i < (int)shape_.size(); i++) inner_size *= shape_[i];
    
    std::vector<float> host_data = getData();
    std::vector<float> permuted_data(size_);
    
    for (int outer = 0; outer < outer_size; outer++) {
        for (int seq = 0; seq < seq_len; seq++) {
            int source_seq = (seq % n) * d + (seq / n);
            
            for (int inner = 0; inner < inner_size; inner++) {
                int src_idx = outer * seq_len * inner_size + source_seq * inner_size + inner;
                int dst_idx = outer * seq_len * inner_size + seq * inner_size + inner;
                permuted_data[dst_idx] = host_data[src_idx];
            }
        }
    }
    
    cudaMemcpyAsync(tensor_.data<float>(), permuted_data.data(),
                    size_ * sizeof(float), cudaMemcpyHostToDevice, data_stream_);
    cudaStreamSynchronize(data_stream_);
}

void DTensor::unpermute_striped(int dim) {
    if (dim < 0 || dim >= (int)shape_.size()) {
        throw std::runtime_error("DTensor::unpermute_striped: Invalid dimension");
    }
    
    int seq_len = shape_[dim];
    int d = world_size_;
    int n = seq_len / d;
    
    if (seq_len % d != 0) {
        std::ostringstream oss;
        oss << "DTensor::unpermute_striped: Sequence length (" << seq_len 
            << ") must be divisible by world_size (" << d << ")";
        throw std::runtime_error(oss.str());
    }
    
    int outer_size = 1;
    int inner_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= shape_[i];
    for (int i = dim + 1; i < (int)shape_.size(); i++) inner_size *= shape_[i];
    
    std::vector<float> host_data = getData();
    std::vector<float> unpermuted_data(size_);
    
    for (int outer = 0; outer < outer_size; outer++) {
        for (int seq = 0; seq < seq_len; seq++) {
            int source_seq = (seq % d) * n + (seq / d);
            
            for (int inner = 0; inner < inner_size; inner++) {
                int src_idx = outer * seq_len * inner_size + source_seq * inner_size + inner;
                int dst_idx = outer * seq_len * inner_size + seq * inner_size + inner;
                unpermuted_data[dst_idx] = host_data[src_idx];
            }
        }
    }
    
    cudaMemcpyAsync(tensor_.data<float>(), unpermuted_data.data(),
                    size_ * sizeof(float), cudaMemcpyHostToDevice, data_stream_);
    cudaStreamSynchronize(data_stream_);
}

// ============================================================================
// Assemble/Gather (from _adhi_)
// ============================================================================

void DTensor::assemble(int dim, int root, DTensor& sharded_tensor) {
    // Gather sharded tensor pieces back to full tensor
    std::vector<int64_t> global_shape = sharded_tensor.get_layout().get_global_shape();
    size_t total_numel = numelFromShape(global_shape);
    
    // Ensure this tensor has space for full assembled result
    if (tensor_.numel() != total_numel) {
        tensor_ = OwnTensor::Tensor(toShape(global_shape),
                                    OwnTensor::TensorOptions()
                                        .with_device(tensor_.device())
                                        .with_dtype(tensor_.dtype()));
    }
    
    // All-gather the sharded pieces
    size_t shard_numel = sharded_tensor.local_tensor().numel();
    pg_->all_gather(sharded_tensor.local_tensor().data<float>(), 
                    tensor_.data<float>(),
                    shard_numel, OwnTensor::Dtype::Float32, true);
    
    layout_ = Layout::replicated(*device_mesh_, std::vector<int64_t>(global_shape.begin(), global_shape.end()));
    shape_ = global_shape;
    size_ = total_numel;
}

// ============================================================================
// Linear Layer Operations (from _adhi_)
// ============================================================================

void DTensor::Linear(DTensor& Input, DTensor& Weights, DTensor& Bias) {
    // Compute Y = matmul(Input, Weights) + Bias
    OwnTensor::Tensor matmul_result = Bridge::matmul(Input.tensor_, Weights.tensor_);
    tensor_ = Bridge::add(matmul_result, Bias.tensor_);
}

void DTensor::linear_w_autograd(DTensor& Input, DTensor& Weights, DTensor& Bias) {
    // Use autograd-aware operations for gradient tracking
    OwnTensor::Tensor out = Bridge::autograd::matmul(Input.tensor_, Weights.tensor_);
    tensor_ = Bridge::autograd::add(out, Bias.tensor_);
}

DTensor DTensor::layer_norm(const DTensor& weight, const DTensor& bias, float eps) const {
    // Current assumption: weights and bias are replicated across TP ranks.
    // Each rank performs local LayerNorm on its shard of activation tokens.
    // Normalized shape is deduced from weight's global shape.
    int normalized_shape = (int)weight.shape()[0];
    
    OwnTensor::Tensor result_local = Bridge::autograd::layer_norm(
        this->tensor_, weight.tensor_, bias.tensor_, normalized_shape, eps);
        
    return DTensor(device_mesh_, pg_, result_local, this->layout_);
}


// ============================================================================
// Distributed Cross Entropy Implementation
// ============================================================================

class DistributedSparseCrossEntropyBackward : public OwnTensor::Node {
    std::shared_ptr<DeviceMesh> mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    OwnTensor::Tensor logits_shard_;
    OwnTensor::Tensor targets_;
    OwnTensor::Tensor max_logits_;
    OwnTensor::Tensor sum_exps_;
    int64_t vocab_offset_;
    int64_t num_classes_local_;

public:
    DistributedSparseCrossEntropyBackward(
        std::shared_ptr<DeviceMesh> mesh,
        std::shared_ptr<ProcessGroupNCCL> pg,
        OwnTensor::Tensor logits_shard,
        OwnTensor::Tensor targets,
        OwnTensor::Tensor max_logits,
        OwnTensor::Tensor sum_exps,
        int64_t vocab_offset,
        int64_t num_classes_local)
        : Node(1),
          mesh_(mesh),
          pg_(pg),
          logits_shard_(logits_shard),
          targets_(targets),
          max_logits_(max_logits),
          sum_exps_(sum_exps),
          vocab_offset_(vocab_offset),
          num_classes_local_(num_classes_local) {}

    OwnTensor::variable_list apply(OwnTensor::variable_list&& grads) override {
        OwnTensor::Tensor grad_output = grads[0];
        
        OwnTensor::Tensor grad_logits_shard = OwnTensor::Tensor::empty(logits_shard_.shape(), 
                                                                    OwnTensor::TensorOptions()
                                                                        .with_dtype(logits_shard_.dtype())
                                                                        .with_device(logits_shard_.device()));
        
        int64_t batch_size = targets_.numel();
        cudaStream_t stream = 0; // Use default for now or shared stream

        if (logits_shard_.device().is_cuda()) {
            if (targets_.dtype() == OwnTensor::Dtype::UInt16) {
                DistributedLoss::launch_distributed_sparse_ce_backward<float, uint16_t>(
                    grad_logits_shard.data<float>(),
                    logits_shard_.data<float>(),
                    targets_.data<uint16_t>(),
                    max_logits_.data<float>(),
                    sum_exps_.data<float>(),
                    grad_output.data<float>(),
                    batch_size,
                    num_classes_local_,
                    vocab_offset_,
                    stream
                );
            } else if (targets_.dtype() == OwnTensor::Dtype::Int64) {
                 DistributedLoss::launch_distributed_sparse_ce_backward<float, int64_t>(
                    grad_logits_shard.data<float>(),
                    logits_shard_.data<float>(),
                    targets_.data<int64_t>(),
                    max_logits_.data<float>(),
                    sum_exps_.data<float>(),
                    grad_output.data<float>(),
                    batch_size,
                    num_classes_local_,
                    vocab_offset_,
                    stream
                );
            }
        }
        
        return {grad_logits_shard};
    }

    std::string name() const override { return "DistributedSparseCrossEntropyBackward"; }
};

DTensor DTensor::distributed_sparse_cross_entropy_loss(const DTensor& target) const {
    if (!layout_.is_sharded()) {
         // Fallback to normal loss if not sharded
         return this->sparse_cross_entropy_loss(target);
    }
    
    int shard_dim = layout_.get_shard_dim();
    if (shard_dim != (int)layout_.get_global_shape().size() - 1) {
        throw std::runtime_error("distributed_sparse_cross_entropy_loss: currently only supports sharding along the last dimension (vocab)");
    }
    
    OwnTensor::Tensor logits_shard = this->tensor_;
    OwnTensor::Tensor targets = target.local_tensor();
    
    int64_t batch_size = targets.numel();
    int64_t num_classes_local = logits_shard.shape().dims.back();
    int64_t vocab_offset = layout_.get_local_offset(pg_->get_rank());
    
    // 1. Local Max
    OwnTensor::Tensor local_max = OwnTensor::reduce_max(logits_shard, {(int64_t)logits_shard.ndim() - 1}, true);
    
    // 2. Global Max
    OwnTensor::Tensor global_max = OwnTensor::Tensor::empty(local_max.shape(), local_max.opts());
    pg_->all_reduce(local_max.data(), global_max.data(), local_max.numel(), local_max.dtype(), op_t::max, false);
    
    // 3. Local SumExp
    // Use manual exp(logits - max) to avoid creating too many large temporaries if possible,
    // although exp(logits_shard) is same size as logits_shard.
    OwnTensor::Tensor exp_shard = OwnTensor::exp(logits_shard - global_max);
    OwnTensor::Tensor local_sum_exp = OwnTensor::reduce_sum(exp_shard, {(int64_t)logits_shard.ndim() - 1}, true);
    
    // 4. Global SumExp
    OwnTensor::Tensor global_sum_exp = OwnTensor::Tensor::empty(local_sum_exp.shape(), local_sum_exp.opts());
    pg_->all_reduce(local_sum_exp.data(), global_sum_exp.data(), local_sum_exp.numel(), local_sum_exp.dtype(), op_t::sum, false);
    
    // 5. Target Logit extraction (from shard)
    OwnTensor::Tensor target_logits_shard = OwnTensor::Tensor::zeros(OwnTensor::Shape{{batch_size, 1}}, logits_shard.opts());
    
    if (targets.dtype() == OwnTensor::Dtype::UInt16) {
        DistributedLoss::launch_distributed_sparse_ce_forward<float, uint16_t>(
            logits_shard.data<float>(),
            targets.data<uint16_t>(),
            target_logits_shard.data<float>(),
            batch_size,
            num_classes_local,
            vocab_offset
        );
    } else if (targets.dtype() == OwnTensor::Dtype::Int64) {
         DistributedLoss::launch_distributed_sparse_ce_forward<float, int64_t>(
            logits_shard.data<float>(),
            targets.data<int64_t>(),
            target_logits_shard.data<float>(),
            batch_size,
            num_classes_local,
            vocab_offset
        );
    }
    
    // 6. Global Target Logit
    OwnTensor::Tensor global_target_logits = OwnTensor::Tensor::empty(target_logits_shard.shape(), target_logits_shard.opts());
    pg_->all_reduce(target_logits_shard.data(), global_target_logits.data(), target_logits_shard.numel(), target_logits_shard.dtype(), op_t::sum, false);
    
    // 7. Loss calculation
    OwnTensor::Tensor sample_losses = OwnTensor::log(global_sum_exp) + global_max - global_target_logits;
    OwnTensor::Tensor mean_loss = OwnTensor::reduce_mean(sample_losses);
    
    // 8. Autograd setup
    if (this->requires_grad()) {
        auto grad_fn = std::make_shared<DistributedSparseCrossEntropyBackward>(
            device_mesh_, pg_, logits_shard, targets, global_max, global_sum_exp, vocab_offset, num_classes_local
        );
        
        // Connect backward node
        OwnTensor::Tensor& mutable_loss = const_cast<OwnTensor::Tensor&>(mean_loss);
        mutable_loss.set_grad_fn(grad_fn);
        mutable_loss.set_requires_grad(true);
        
        // Link to logits_shard
        OwnTensor::Tensor& mutable_logits = const_cast<OwnTensor::Tensor&>(logits_shard);
        grad_fn->set_next_edge(0, OwnTensor::autograd::get_grad_edge(mutable_logits));
    }
    
    return DTensor(device_mesh_, pg_, mean_loss, Layout::replicated(*device_mesh_, {1}));
}
