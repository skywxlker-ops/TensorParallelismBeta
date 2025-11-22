#include "tensor/dtensor.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <numeric> // For std::accumulate
#include <stdexcept> // For std::runtime_error
#include <sstream>

// Global allocator definition
CachingAllocator gAllocator;

// =========================================================
// Constructor / Destructor
// =========================================================

// Main public constructor (Initializes an empty/default tensor)
DTensor::DTensor(std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroup> pg)
    : rank_(pg->getRank()),
      world_size_(pg->getWorldSize()),
      mesh_(mesh),
      pg_(pg),
      stream_(pg->getStream()),
      layout_(Layout(mesh, {}, ShardingType::REPLICATED)), // Default layout
      size_(0),
      shape_({0}),
      tensor_(OwnTensor::Shape{{1}}, // Default empty tensor wrapper
              OwnTensor::TensorOptions()
                  .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
                  .with_dtype(OwnTensor::Dtype::Float32)),
      temp_tensor_(tensor_) { // Initialize temp_tensor_
    cudaSetDevice(rank_);
}

// Private constructor (Used for internal op results)
DTensor::DTensor(std::shared_ptr<DeviceMesh> mesh,
                 std::shared_ptr<ProcessGroup> pg,
                 const OwnTensor::Tensor& local_tensor,
                 const Layout& layout)
    : rank_(pg->getRank()),
      world_size_(pg->getWorldSize()),
      mesh_(mesh),
      pg_(pg),
      stream_(pg->getStream()),
      layout_(layout),
      tensor_(local_tensor),           
      temp_tensor_(local_tensor)      
{
    cudaSetDevice(rank_);
    
    // Calculate local shape and size from the new layout
    shape_ = layout_.get_local_shape(rank_); // shape_ is the local shape
    size_ = 1;
    for (int d : shape_) size_ *= d;
}

DTensor::~DTensor() {
    // Ensure stream is done before freeing memory
    cudaStreamSynchronize(stream_);
}

// =========================================================
// Data Setup
// =========================================================
void DTensor::setData(const std::vector<float>& host_data, const Layout& layout) {
    layout_ = layout;
    mesh_ = layout_.get_mesh(); // Ensure mesh is in sync
    
    std::vector<int> local_shape = layout_.get_local_shape(rank_);
    shape_ = local_shape; // Update legacy local shape member
    
    // Calculate local size
    size_ = 1;
    for (int d : local_shape) size_ *= d;

    if (host_data.size() != (size_t)size_) {
        std::ostringstream oss;
        oss << "DTensor::setData: host_data size (" << host_data.size() 
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

    // Re-allocate temp tensor buffer for new shape
    temp_tensor_ = OwnTensor::Tensor(shape_obj, opts);
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
    // Performs an inplace AllReduce (Sum) on the local tensor data
    auto work = pg_->allReduce<float>(tensor_.data<float>(), size_, ncclFloat);
    work->wait();
}

void DTensor::reduceScatter() {
    // This assumes tensor_ holds the full data to be scattered
    // and temp_tensor_ is the receiver for the local shard.
    auto work = pg_->reduceScatter<float>(
        temp_tensor_.data<float>(), tensor_.data<float>(), size_ / world_size_, ncclFloat);
    work->wait();
    
    // Swap buffers so tensor_ now holds the local shard
    std::swap(tensor_, temp_tensor_);
    shape_ = layout_.get_local_shape(rank_);
    size_ = 1; for(int d : shape_) size_ *= d;
}

void DTensor::allGather() {
    // Gathers local tensors (tensor_) from all ranks into temp_tensor_
    // Assumes temp_tensor_ is large enough (global size)
    auto work = pg_->allGather<float>(
        temp_tensor_.data<float>(), tensor_.data<float>(), size_, ncclFloat);
    work->wait();
    
    // Swap buffers so tensor_ now holds the full global data
    std::swap(tensor_, temp_tensor_);
    shape_ = layout_.get_global_shape();
    size_ = 1; for(int d : shape_) size_ *= d;
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
    if (!layout_.is_compatible(other.get_layout())) { \
        throw std::runtime_error("Incompatible layouts for operation " #op_name); \
    } \
    OwnTensor::Tensor result = TensorOpsBridge::op_name(tensor_, other.tensor_); \
    /* Element-wise ops preserve layout */ \
    return DTensor(mesh_, pg_, result, layout_); \
}

DEFINE_TENSOR_OP(add, add)
DEFINE_TENSOR_OP(sub, sub)
DEFINE_TENSOR_OP(mul, mul)
DEFINE_TENSOR_OP(div, div)

// =========================================================
// Distributed Matmul (The Core Logic)
// =========================================================

DTensor DTensor::matmul(const DTensor& other) const {
    const Layout& a_layout = this->layout_;
    const Layout& b_layout = other.get_layout();

    // --- CASE 1: Column-Parallel Matmul ---
    // Typically Layer 1 of MLP.
    // Identity: X [M, K] @ W [K, N/P] -> Y [M, N/P]
    // Condition: A is Replicated, B is Sharded on Dim 1 (Columns)
    if (a_layout.is_replicated() && b_layout.is_sharded_by_dim(1)) {
        return _column_parallel_matmul(other);
    }
    
    // --- CASE 2: Row-Parallel Matmul (Split-K) ---
    // Typically Layer 2 of MLP.
    // Identity: X [M, K/P] @ W [K/P, N] -> Y_partial [M, N] -> AllReduce -> Y [M, N]
    // Condition: A is Sharded on Dim 1 (Columns of prev layer), 
    //            B is Sharded on Dim 0 (Rows).
    if (a_layout.is_sharded_by_dim(1) && b_layout.is_sharded_by_dim(0)) {
         return _row_parallel_matmul(other);
    }

    // --- Fallback / Error ---
    std::ostringstream oss;
    oss << "DTensor::matmul: This sharding combination is not implemented!\n"
        << "  Layout A: " << a_layout.describe(rank_) << "\n"
        << "  Layout B: " << b_layout.describe(rank_);
    throw std::runtime_error(oss.str());
}

// Private helper for: Y_col_shard = X_replicated @ W_col_shard
DTensor DTensor::_column_parallel_matmul(const DTensor& other) const {
    // 1. Perform local matmul
    //    A = [M, K], B = [K, N_shard] -> Y_shard = [M, N_shard]
    OwnTensor::Tensor Y_shard = TensorOpsBridge::matmul(this->tensor_, other.local_tensor());

    // 2. Calculate output global shape: [M, N_global]
    std::vector<int> Y_global_shape = {
        this->layout_.get_global_shape()[0],
        other.get_layout().get_global_shape()[1]
    };
    
    // 3. Calculate output layout: sharded by column (dim 1)
    Layout Y_layout(mesh_, Y_global_shape, ShardingType::SHARDED, 1);

    // 4. Return new DTensor (No comms needed)
    return DTensor(mesh_, pg_, Y_shard, Y_layout);
}

// Private helper for: Y_replicated = AllReduce(X_shard @ W_row_shard)
DTensor DTensor::_row_parallel_matmul(const DTensor& other) const {
    // 1. Perform local matmul
    //    A = [M, K_shard], B = [K_shard, N] -> Y_partial = [M, N]
    //    This produces a Partial Sum result on each rank.
    OwnTensor::Tensor Y_partial = TensorOpsBridge::matmul(this->tensor_, other.local_tensor());

    // 2. The *global* output shape is [M, N]
    std::vector<int> Y_global_shape = {
        this->layout_.get_global_shape()[0],
        other.get_layout().get_global_shape()[1]
    };
    
    // 3. The final layout will be REPLICATED (after reduction).
    Layout Y_layout(mesh_, Y_global_shape, ShardingType::REPLICATED);
    
    // 4. Create the output DTensor. It holds the partial data initially.
    DTensor Y_out(mesh_, pg_, Y_partial, Y_layout);
    
    // 5. Perform AllReduce to sum the partial results from all ranks.
    //    After this, every rank holds the fully summed, replicated tensor.
    Y_out.allReduce(); 
    
    return Y_out;
}


// =========================================================
// Redistribute
// =========================================================
DTensor DTensor::redistribute(const Layout& new_layout) const {
    // Case 1: Current layout is sharded, new layout is replicated
    if (!layout_.is_replicated() && new_layout.is_replicated()) {
        auto local_shape = layout_.get_local_shape(rank_);
        size_t local_numel = tensor_.numel();
        
        OwnTensor::Shape global_shape_obj;
        global_shape_obj.dims.assign(new_layout.get_global_shape().begin(), new_layout.get_global_shape().end());
        OwnTensor::Tensor gathered_tensor(global_shape_obj, OwnTensor::TensorOptions()
                                                                    .with_device(tensor_.device())
                                                                    .with_dtype(tensor_.dtype()));

        if (layout_.get_shard_dim() == 0) {
            // Sharded on rows: data is already in the correct global order
            pg_->allGather<float>(
                gathered_tensor.data<float>(),
                const_cast<float*>(this->tensor_.data<float>()),
                local_numel,
                ncclFloat
            )->wait();
        } else if (layout_.get_shard_dim() == 1) {
            // Sharded on columns: requires reordering after gather
            float* tmp_buffer;
            cudaMalloc(&tmp_buffer, local_numel * world_size_ * sizeof(float));

            pg_->allGather<float>(
                tmp_buffer,
                const_cast<float*>(this->tensor_.data<float>()),
                local_numel,
                ncclFloat
            )->wait();

            for (int i = 0; i < world_size_; ++i) {
                cudaMemcpy2DAsync(
                    gathered_tensor.data<float>() + i * local_shape[1],
                    new_layout.get_global_shape()[1] * sizeof(float),
                    tmp_buffer + i * local_numel,
                    local_shape[1] * sizeof(float),
                    local_shape[1] * sizeof(float),
                    local_shape[0],
                    cudaMemcpyDeviceToDevice,
                    stream_
                );
            }
            cudaFree(tmp_buffer);
        }
        return DTensor(mesh_, pg_, gathered_tensor, new_layout);
    }

    // Case 2: Current layout is replicated, new layout is sharded
    if (layout_.is_replicated() && !new_layout.is_replicated()) {
        auto new_local_shape_vec = new_layout.get_local_shape(rank_);
        OwnTensor::Shape new_local_shape_obj;
        new_local_shape_obj.dims.assign(new_local_shape_vec.begin(), new_local_shape_vec.end());
        
        OwnTensor::Tensor sliced_tensor(new_local_shape_obj, OwnTensor::TensorOptions()
                                                                .with_device(tensor_.device())
                                                                .with_dtype(tensor_.dtype()));

        if (new_layout.get_shard_dim() == 0) {
            // Sharding on rows: slice is a contiguous block
            size_t slice_size = sliced_tensor.numel();
            size_t offset = rank_ * slice_size;
            cudaMemcpyAsync(
                sliced_tensor.data<float>(),
                this->tensor_.data<float>() + offset,
                slice_size * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream_
            );
        } else if (new_layout.get_shard_dim() == 1) {
            // Sharding on columns: slice is not contiguous, requires 2D copy
            size_t src_pitch = layout_.get_global_shape()[1] * sizeof(float);
            size_t dst_pitch = new_local_shape_vec[1] * sizeof(float);
            size_t copy_width_bytes = new_local_shape_vec[1] * sizeof(float);

            cudaMemcpy2DAsync(
                sliced_tensor.data<float>(),
                dst_pitch,
                this->tensor_.data<float>() + (rank_ * new_local_shape_vec[1]),
                src_pitch,
                copy_width_bytes,
                new_local_shape_vec[0],
                cudaMemcpyDeviceToDevice,
                stream_
            );
        }
        return DTensor(mesh_, pg_, sliced_tensor, new_layout);
    }

    // Case 3: Sharded to Sharded
    if (layout_ != new_layout) {
        DTensor replicated_tensor = this->redistribute(
            Layout(mesh_, layout_.get_global_shape(), ShardingType::REPLICATED)
        );
        return replicated_tensor.redistribute(new_layout);
    }

    // If layouts are the same, just return a copy
    return *this;
}


// =========================================================
// Reshape
// =========================================================
DTensor DTensor::reshape(const std::vector<int>& new_global_shape) const {
    // 1. Ask the layout to calculate the new sharding
    Layout new_layout = layout_.reshape(new_global_shape);

    // 2. Get the new local shape from the new layout
    std::vector<int> new_local_shape = new_layout.get_local_shape(rank_);
    
    int new_local_size = 1;
    for (int d : new_local_shape) new_local_size *= d;
    if (new_local_size != size_) {
        throw std::runtime_error("DTensor::reshape: local element count mismatch. Reshaping sharded tensors is complex.");
    }

    OwnTensor::Shape shape_obj;
    shape_obj.dims.assign(new_local_shape.begin(), new_local_shape.end());
    
    // 3. Perform the local reshape
    OwnTensor::Tensor reshaped_tensor = tensor_.reshape(shape_obj);

    // 4. Return the new DTensor
    return DTensor(mesh_, pg_, reshaped_tensor, new_layout);
}


// =========================================================
// Checkpointing (N-D Safe)
// =========================================================
// This saves the *local* data.
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

    int ndim = static_cast<int>(shape_.size()); // shape_ is local shape
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
    
    std::vector<int> loaded_shape(ndim);
    file.read(reinterpret_cast<char*>(loaded_shape.data()), ndim * sizeof(int));
    char dtype_buf[32];
    file.read(dtype_buf, sizeof(dtype_buf));
    dtype_ = std::string(dtype_buf);
    
    int loaded_size = 1;
    for (int d : loaded_shape) loaded_size *= d;

    std::vector<float> host_data(loaded_size);
    file.read(reinterpret_cast<char*>(host_data.data()), loaded_size * sizeof(float));
    file.close();

    // WARNING: This checkpointing method does not save the global layout.
    // We are forced to assume the loaded tensor is REPLICATED,
    // with its global shape being equal to the local shape we just read.
    Layout loaded_layout(mesh_, loaded_shape, ShardingType::REPLICATED);
    
    setData(host_data, loaded_layout);

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
    if (dims.empty() || dim < 0) return;

    if ((size_t)dim == dims.size() - 1) {
        std::cout << "[";
        for (int i = 0; i < dims[dim]; ++i) {
            if (i > 0) std::cout << ", ";
            if ((size_t)(offset + i) < data.size()) {
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

void DTensor::print() const {
    if (size_ <= 0 || tensor_.data<float>() == nullptr) {
        std::cerr << "[Rank " << rank_ << "] Tensor is empty or uninitialized!\n";
        return;
    }

    // Print layout information
    std::cout << layout_.describe(rank_) << std::endl;

    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(),
                    tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronize(stream_);

    std::cout << "[Rank " << rank_ << " Data] ";
    printRecursive(host_data, shape_, 0, 0); // shape_ is local shape
    std::cout << "\n";
}
