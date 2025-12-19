#include "tensor/dtensor.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <numeric> 
#include <stdexcept> 
#include <sstream>


CachingAllocator gAllocator;


DTensor::DTensor(std::shared_ptr<DeviceMesh> device_mesh, std::shared_ptr<ProcessGroupNCCL> pg, std::shared_ptr<StreamPool> stream_pool)
    : rank_(pg->get_rank()),
      world_size_(pg->get_worldsize()),
      device_mesh_(device_mesh),
      pg_(pg),
      stream_pool_(stream_pool ? stream_pool : std::make_shared<StreamPool>(pg->get_local_rank(), 4)),
      compute_stream_(stream_pool_ ? stream_pool_->getComputeStream() : nullptr),
      comm_stream_(stream_pool_ ? stream_pool_->getCommStream() : nullptr),
      data_stream_(stream_pool_ ? stream_pool_->getDataStream() : nullptr),
      layout_(Layout::replicated(device_mesh, {})), 
      size_(0),
      shape_({0}),
      tensor_(OwnTensor::Shape{{1}},
              OwnTensor::TensorOptions()
                  .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
                  .with_dtype(OwnTensor::Dtype::Float32)),
      temp_tensor_(tensor_) { // TEMPORARY: Restored for debugging
    cudaSetDevice(pg->get_local_rank());
    
    // Create events for stream synchronization (enable overlap)
    cudaEventCreateWithFlags(&compute_event_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&comm_event_, cudaEventDisableTiming);
}

DTensor::DTensor(std::shared_ptr<DeviceMesh> device_mesh,
                 std::shared_ptr<ProcessGroupNCCL> pg,
                 const OwnTensor::Tensor& local_tensor,
                 const Layout& layout)
    : rank_(pg->get_rank()),
      world_size_(pg->get_worldsize()),
      device_mesh_(device_mesh),
      pg_(pg),
      stream_pool_(std::make_shared<StreamPool>(pg->get_local_rank(), 4)),
      compute_stream_(stream_pool_->getComputeStream()),
      comm_stream_(stream_pool_->getCommStream()),
      data_stream_(stream_pool_->getDataStream()),
      layout_(layout),
      tensor_(local_tensor),
      temp_tensor_(local_tensor)  // TEMPORARY: Restored for debugging
{
    cudaSetDevice(pg->get_local_rank());
    
    shape_ = layout_.get_local_shape(rank_); 
    size_ = 1;
    for (int d : shape_) size_ *= d;
    
    // Create events for stream synchronization (enable overlap)
    cudaEventCreateWithFlags(&compute_event_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&comm_event_, cudaEventDisableTiming);
}

DTensor::~DTensor() {
    // Synchronize all streams to ensure operations complete
    cudaStreamSynchronize(compute_stream_);
    cudaStreamSynchronize(comm_stream_);
    cudaStreamSynchronize(data_stream_);
    
    // Destroy events
    cudaEventDestroy(compute_event_);
    cudaEventDestroy(comm_event_);
}

void DTensor::setData(const std::vector<float>& host_data, const Layout& layout) {
    layout_ = layout;

    std::vector<int> local_shape = layout_.get_local_shape(rank_);
    shape_ = local_shape; 

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
    tensor_.set_data(host_data);  // Uses internal stream, will sync later
    
    // No temp_tensor_ allocation needed
}

// GPU-Native Initialization: Load on root GPU, distribute via NCCL
void DTensor::setDataFromRoot(const std::vector<float>& host_data, 
                                const Layout& layout, int root) {
    layout_ = layout;
    
    std::vector<int> global_shape = layout.get_global_shape();
    std::vector<int> local_shape = layout.get_local_shape(rank_);
    
    size_t global_size = 1;
    for (int d : global_shape) global_size *= d;
    
    size_t local_size = 1;
    for (int d : local_shape) local_size *= d;
    
    // Validate input on root rank
    if (rank_ == root && host_data.size() != global_size) {
        std::ostringstream oss;
        oss << "DTensor::setDataFromRoot: Root rank " << root 
            << " host_data size (" << host_data.size() 
            << ") does not match global tensor size (" << global_size << ").";
        throw std::runtime_error(oss.str());
    }
    
    // Case 1: Replicated layout - Broadcast full tensor directly
    if (layout.is_replicated()) {
        OwnTensor::Shape shape_obj;
        shape_obj.dims.assign(global_shape.begin(), global_shape.end());
        
        OwnTensor::TensorOptions opts;
        opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
                   .with_dtype(OwnTensor::Dtype::Float32);
        
        tensor_ = OwnTensor::Tensor(shape_obj, opts);
        
        // Root loads data to GPU using data stream
        if (rank_ == root) {
            tensor_.set_data(host_data);  // CPU → GPU on root
        }
        
        // Broadcast to all ranks using comm stream (in-place: input == output)
        pg_->broadcast(tensor_.data<float>(), tensor_.data<float>(), global_size, OwnTensor::Dtype::Float32, root, true);
        
        shape_ = global_shape;
        size_ = global_size;
        // No temp_tensor_ allocation
        return;
    }
    
    // Case 2: Sharded layout - Use scatter for memory efficiency
    OwnTensor::TensorOptions opts;
    opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
               .with_dtype(OwnTensor::Dtype::Float32);
    
    // Allocate local shard only (no temp full tensor!)
    OwnTensor::Shape local_shape_obj;
    local_shape_obj.dims.assign(local_shape.begin(), local_shape.end());
    tensor_ = OwnTensor::Tensor(local_shape_obj, opts);
    
    if (layout.get_shard_dim() == 0) {
        // Row-sharded: Use scatter directly (contiguous chunks)
        OwnTensor::Tensor* temp_full_ptr = nullptr;
        if (rank_ == root) {
            // Root loads full tensor
            OwnTensor::Shape global_shape_obj;
            global_shape_obj.dims.assign(global_shape.begin(), global_shape.end());
            temp_full_ptr = new OwnTensor::Tensor(global_shape_obj, opts);
            temp_full_ptr->set_data(host_data);  // CPU → GPU on root
        }
        
        // Scatter: root sends different chunks to each rank (using comm stream)
        pg_->scatter(
            temp_full_ptr ? temp_full_ptr->data<float>() : nullptr,  // send_buf (root only)
            tensor_.data<float>(),                                     // recv_buf (all ranks)
            local_size,                                                // recv_count
            OwnTensor::Dtype::Float32,
            root,
            true
        );
        
        if (temp_full_ptr) delete temp_full_ptr;
        
    } else if (layout.get_shard_dim() == 1) {
        // Column-sharded: Need broadcast then 2D extract (non-contiguous)
        OwnTensor::Shape global_shape_obj;
        global_shape_obj.dims.assign(global_shape.begin(), global_shape.end());
        OwnTensor::Tensor temp_full(global_shape_obj, opts);
        
        if (rank_ == root) {
            temp_full.set_data(host_data);
        }
        
        pg_->broadcast(temp_full.data<float>(), temp_full.data<float>(), global_size, OwnTensor::Dtype::Float32, root, true);
        _extract_local_shard(temp_full, layout);
    }
    
    // Update metadata
    shape_ = local_shape;
    size_ = local_size;
    
    // No pre-allocated temp_tensor_ - will use dynamic allocation when needed
}

// Helper: Extract Local Shard from Full Tensor
void DTensor::_extract_local_shard(const OwnTensor::Tensor& full_tensor, 
                                    const Layout& layout) {
    std::vector<int> local_shape = layout.get_local_shape(rank_);
    std::vector<int> global_shape = layout.get_global_shape();
    
    OwnTensor::Shape shape_obj;
    shape_obj.dims.assign(local_shape.begin(), local_shape.end());
    
    OwnTensor::TensorOptions opts;
    opts = opts.with_device(full_tensor.device())
               .with_dtype(full_tensor.dtype());
    
    tensor_ = OwnTensor::Tensor(shape_obj, opts);
    
    size_t local_size = tensor_.numel();
    
    if (layout.get_shard_dim() == 0) {
        // Row sharding: contiguous copy
        // Calculate offset based on rank
        size_t offset = 0;
        for (int r = 0; r < rank_; ++r) {
            std::vector<int> rank_shape = layout.get_local_shape(r);
            size_t rank_local_size = 1;
            for (int d : rank_shape) rank_local_size *= d;
            offset += rank_local_size;
        }
        
        cudaMemcpyAsync(tensor_.data<float>(), 
                       full_tensor.data<float>() + offset,
                       local_size * sizeof(float),
                       cudaMemcpyDeviceToDevice, data_stream_);
        
    } else if (layout.get_shard_dim() == 1) {
        // Column sharding: non-contiguous 2D slice
        // For a 2D tensor [M, N] sharded on dim 1:
        // - Each rank gets [M, N/world_size]
        
        int rows = global_shape[0];
        int global_cols = global_shape[1];
        int local_cols = local_shape[1];
        int col_offset = rank_ * local_cols;
        
        // Manual 2D column slice extraction
        // Copy each row's column segment from full tensor to local shard
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


std::vector<float> DTensor::getData() const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, data_stream_);
    cudaStreamSynchronize(data_stream_);
    return host_data;
}

/**
 * ensureTempTensor: Lazy allocation for temp_tensor_
 * Only reallocates if the required size doesn't match current size
 */
void DTensor::ensureTempTensor(const std::vector<int>& shape) {
    size_t required_size = 1;
    for (int d : shape) required_size *= d;
    
    // Only allocate if size changed (lazy allocation)
    if (temp_tensor_.numel() != required_size) {
        OwnTensor::Shape shape_obj;
        shape_obj.dims.assign(shape.begin(), shape.end());
        
        temp_tensor_ = OwnTensor::Tensor(shape_obj,
                                         OwnTensor::TensorOptions()
                                             .with_device(tensor_.device())
                                             .with_dtype(tensor_.dtype()));
    }
    // Else: reuse existing temp_tensor_ (memory saving!)
}

// Stream synchronization helpers for overlap
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

void DTensor::allReduce() {
    // In-place: input == output
    pg_->all_reduce(tensor_.data<float>(), tensor_.data<float>(), size_, OwnTensor::Dtype::Float32, sum, true);
}

void DTensor::reduceScatter() {
    size_t count_per_shard = size_ / world_size_;
    
    // ProcessGroup::reduceScatter is in-place: reduces full tensor into specific offsets
    // Input: tensor_ (full data)
    // Output: tensor_ + rank * count (reduced shard)
    pg_->reduce_scatter(
        tensor_.data<float>(), tensor_.data<float>(), count_per_shard, OwnTensor::Dtype::Float32, sum, true);

    // Lazy allocation: only reallocate if size changed
    std::vector<int> local_shape = layout_.get_local_shape(rank_);
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
    size_t count_per_rank = size_; // Current tensor is a shard
    
    // Lazy allocation: only reallocate if size changed
    std::vector<int> global_shape = layout_.get_global_shape();
    ensureTempTensor(global_shape);

    // NCCL allGather: each rank contributes their shard, NCCL gathers into temp_tensor_
    pg_->all_gather(
        tensor_.data<float>(),
        temp_tensor_.data<float>(),
        count_per_rank,
        OwnTensor::Dtype::Float32,
        true
    );
   
    std::swap(tensor_, temp_tensor_);
    shape_ = global_shape;
    size_ = tensor_.numel();
}

void DTensor::broadcast(int root) {
    pg_->broadcast(tensor_.data<float>(), tensor_.data<float>(), size_, OwnTensor::Dtype::Float32, root, true);
}


/**
 * Replicate (in-place) - Broadcast tensor from root to all GPUs
 * Modifies this DTensor to have replicated layout with same tensor on all GPUs
 */
void DTensor::replicate(int root) {
    std::vector<int> global_shape = layout_.get_global_shape();
    size_t total_numel = 1;
    for (int d : global_shape) total_numel *= d;
    
    if (tensor_.numel() != total_numel) {
        OwnTensor::Shape global_shape_obj;
        global_shape_obj.dims.assign(global_shape.begin(), global_shape.end());
        tensor_ = OwnTensor::Tensor(global_shape_obj, 
                                    OwnTensor::TensorOptions()
                                        .with_device(tensor_.device())
                                        .with_dtype(tensor_.dtype()));
    }
    
    pg_->broadcast(tensor_.data<float>(), tensor_.data<float>(), total_numel, OwnTensor::Dtype::Float32, root, true);
    
    layout_ = Layout::replicated(device_mesh_, global_shape);
    shape_ = global_shape;
    size_ = total_numel;
}

/**
 * Shard (in-place) - Distribute tensor across GPUs along specified dimension
 * Modifies this DTensor to have sharded layout along specified dimension
 * 
 * Memory note: Due to ncclScatter writing to buffer offsets, we need a temporary
 * buffer. This is a limitation of the NCCL scatter API.
 */
void DTensor::shard(int dim, int root) {
    std::vector<int> global_shape = layout_.get_global_shape();
    
    if (dim < 0 || dim >= (int)global_shape.size()) {
        std::ostringstream oss;
        oss << "DTensor::shard: Invalid shard dimension " << dim 
            << " for tensor with " << global_shape.size() << " dimensions";
        throw std::runtime_error(oss.str());
    }
    

    Layout sharded_layout(device_mesh_, global_shape, ShardingType::SHARDED, dim);
    std::vector<int> local_shape = sharded_layout.get_local_shape(rank_);
    size_t shard_numel = 1;
    for (int d : local_shape) shard_numel *= d;
    
    pg_->scatter(
        tensor_.data<float>(),
        tensor_.data<float>() + rank_ * shard_numel,  // output offset for this rank
        shard_numel,
        OwnTensor::Dtype::Float32,
        root,
        true
    );
    
    OwnTensor::Shape local_shape_obj;
    local_shape_obj.dims.assign(local_shape.begin(), local_shape.end());
    OwnTensor::Tensor shard_tensor(local_shape_obj,
                                   OwnTensor::TensorOptions()
                                       .with_device(tensor_.device())
                                       .with_dtype(tensor_.dtype()));
    
    // Copy shard to new tensor using data stream
    cudaMemcpyAsync(
        shard_tensor.data<float>(),
        tensor_.data<float>() + rank_ * shard_numel,
        shard_numel * sizeof(float),
        cudaMemcpyDeviceToDevice,
        data_stream_
    );
    cudaStreamSynchronize(data_stream_);
    
    std::swap(tensor_, shard_tensor);  // Swap instead of move
    layout_ = sharded_layout;
    shape_ = local_shape;
    size_ = shard_numel;
}

/**
 * Sync (in-place) - Synchronize tensor values across GPUs using AllReduce with SUM
 * Uses stream overlap: communication starts on comm_stream_ while compute_stream_ can continue
 */
void DTensor::sync() {
    // Record that current compute operations are done
    recordComputeDone();
    
    // Make communication stream wait for compute to finish
    waitForCompute();
    
    // Start AllReduce on communication stream (async!)
    pg_->all_reduce(tensor_.data<float>(), tensor_.data<float>(), size_, OwnTensor::Dtype::Float32, sum, true);
    
    // Record that communication is done (for future compute operations that need this result)
    recordCommDone();
    
    // Note: Caller can now use waitForComm() if they need to wait for this AllReduce
    // Otherwise, computation can overlap with communication!
}

/**
 * Scale (in-place) - Multiply all tensor values by a scalar
 */
void DTensor::scale(float factor) {
    // Use element-wise multiplication with scalar
    // This calls the underlying tensor library's scale operation
    OwnTensor::Tensor scaled = TensorOpsBridge::mul(tensor_, factor);
    tensor_ = std::move(scaled);
}

#define DEFINE_TENSOR_OP(func, op_name) \
DTensor DTensor::func(const DTensor& other) const { \
    if (!layout_.is_compatible(other.get_layout())) { \
        throw std::runtime_error("Incompatible layouts for operation " #op_name); \
    } \
    OwnTensor::Tensor result = TensorOpsBridge::op_name(tensor_, other.tensor_); \
    return DTensor(device_mesh_, pg_, result, layout_); \
}

DEFINE_TENSOR_OP(add, add)
DEFINE_TENSOR_OP(sub, sub)
DEFINE_TENSOR_OP(mul, mul)
DEFINE_TENSOR_OP(div, div)



DTensor DTensor::matmul(const DTensor& other) const {
    const Layout& a_layout = this->layout_;
    const Layout& b_layout = other.get_layout();

    auto a_placement = a_layout.get_placement(0); 
    auto b_placement = b_layout.get_placement(0);


    // X [M, K] @ W [K, N/P] -> Y [M, N/P]

    if (a_placement->type() == PlacementType::REPLICATE &&
        b_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(b_placement.get())->dim() == 1) {
        return _column_parallel_matmul(other);
    }

    // X [M, K/P] @ W [K/P, N] -> Y_partial [M, N] -> AllReduce -> Y [M, N]

    if (a_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(a_placement.get())->dim() == 1 &&
        b_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(b_placement.get())->dim() == 0) {
         return _row_parallel_matmul(other);
    }

    std::ostringstream oss;
    oss << "DTensor::matmul: This sharding combination is not implemented!\n"
        << "  Layout A: " << a_layout.describe(rank_) << "\n"
        << "  Layout B: " << b_layout.describe(rank_);
    throw std::runtime_error(oss.str());
}

DTensor DTensor::_column_parallel_matmul(const DTensor& other) const {
    // Column-Parallel: X [M, K] @ W1 [K, N/P] -> H [M, N/P]
    // X is replicated (same on all GPUs)
    // W1 is sharded on dim 1 (each GPU has different columns)
    // Output H is sharded on dim 1 (each GPU has M x N/P)
    
    // Step 1: Local matmul
    OwnTensor::Tensor Y_shard = TensorOpsBridge::matmul(this->tensor_, other.local_tensor());

    // Step 2: Create SHARDED output layout
    std::vector<int> Y_global_shape = {
        this->layout_.get_global_shape()[0],
        other.get_layout().get_global_shape()[1]
    };
    Layout Y_layout(device_mesh_, Y_global_shape, ShardingType::SHARDED, 1);
    
    return DTensor(device_mesh_, pg_, Y_shard, Y_layout);
}

DTensor DTensor::_row_parallel_matmul(const DTensor& other) const {
    // Row-Parallel: H [M, N/P] @ W2 [N/P, K] -> Y_partial [M, K]
    // Each GPU computes partial result, we use SYNC to sum them
    
    // Step 1: Local matmul - produces partial result
    OwnTensor::Tensor Y_partial = TensorOpsBridge::matmul(this->tensor_, other.local_tensor());

    // Step 2: Create replicated layout (all GPUs will have same data after sync)
    std::vector<int> Y_global_shape = {
        this->layout_.get_global_shape()[0],
        other.get_layout().get_global_shape()[1]
    };
    Layout Y_layout = Layout::replicated(device_mesh_, Y_global_shape);
    DTensor Y_out(device_mesh_, pg_, Y_partial, Y_layout);

    // Step 3: Use SYNC primitive (AllReduce with SUM)
    // Y = Y_partial_0 + Y_partial_1 + ... = H_full @ W2_full
    Y_out.sync();
    
    return Y_out;
}

/**
 * Fused matmul + GELU activation
 * Elimin ates intermediate memory read/write by applying GELU in-place after matmul
 */
DTensor DTensor::matmul_gelu(const DTensor& other) const {
    // Perform matmul first (uses TensorOpsBridge which runs on compute stream)
    DTensor result = this->matmul(other);
    
    // Apply GELU activation in-place on the result using compute stream
    launch_activation_kernel(
        result.tensor_.data<float>(),
        result.size_,
        ActivationType::GELU,
        result.compute_stream_  // Use compute stream for kernel execution
    );
    
    // No synchronization here - let caller manage stream sync
    return result;
}

/**
 * Fused matmul + ReLU activation
 * Eliminates intermediate memory read/write by applying ReLU in-place after matmul
 */
DTensor DTensor::matmul_relu(const DTensor& other) const {
    // Perform matmul first (uses TensorOpsBridge which runs on compute stream)
    DTensor result = this->matmul(other);
    
    // Apply ReLU activation in-place on the result using compute stream
    launch_activation_kernel(
        result.tensor_.data<float>(),
        result.size_,
        ActivationType::RELU,
        result.compute_stream_  // Use compute stream for kernel execution
    );
    
    // No synchronization here - let caller manage stream sync
    return result;
}





DTensor DTensor::reshape(const std::vector<int>& new_global_shape) const {

    if (layout_.is_sharded()) {
        throw std::runtime_error("DTensor::reshape: Reshaping sharded tensors not yet implemented");
    }


    int new_local_size = 1;
    for (int d : new_global_shape) new_local_size *= d;
    if (new_local_size != size_) {
        throw std::runtime_error("DTensor::reshape: local element count mismatch.");
    }

    OwnTensor::Shape shape_obj;
    shape_obj.dims.assign(new_global_shape.begin(), new_global_shape.end());
    

    OwnTensor::Tensor reshaped_tensor = tensor_.reshape(shape_obj);


    Layout new_layout = Layout::replicated(device_mesh_, new_global_shape);

    return DTensor(device_mesh_, pg_, reshaped_tensor, new_layout);
}


// Checkpointing (N-D Safe)

void DTensor::saveCheckpoint(const std::string& path) const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, data_stream_);
    cudaStreamSynchronize(data_stream_);

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

    Layout loaded_layout = Layout::replicated(device_mesh_, loaded_shape);
    
    setData(host_data, loaded_layout);

    std::cout << "[Rank " << rank_ << "] Checkpoint loaded: " << path
              << " (" << ndim << "D, " << size_ << " elements)\n";
}

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

    std::cout << layout_.describe(rank_) << std::endl;

    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(),
                    tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    data_stream_);
    cudaStreamSynchronize(data_stream_);

    std::cout << "[Rank " << rank_ << " Data] ";
    printRecursive(host_data, shape_, 0, 0);
    std::cout << "\n";
}
// ============================================================================
// Async Collective Operations (for use within NCCL groups)
// ============================================================================

void DTensor::allReduce_async() {
    recordComputeDone();
    waitForCompute();
    pg_->all_reduce_async(tensor_.data<float>(), tensor_.data<float>(), size_, OwnTensor::Dtype::Float32, sum, false);
    recordCommDone();
}

void DTensor::sync_async() {
    recordComputeDone();
    waitForCompute();
    pg_->all_reduce_async(tensor_.data<float>(), tensor_.data<float>(), size_, OwnTensor::Dtype::Float32, sum, false);
    recordCommDone();
}

void DTensor::reduceScatter_async() {
    size_t count_per_shard = size_ / world_size_;
    recordComputeDone();
    waitForCompute();
    pg_->reduce_scatter_async(tensor_.data<float>(), tensor_.data<float>(), count_per_shard, OwnTensor::Dtype::Float32, sum, false);
    std::vector<int> local_shape = layout_.get_local_shape(rank_);
    ensureTempTensor(local_shape);
    cudaMemcpyAsync(temp_tensor_.data<float>(), tensor_.data<float>() + rank_ * count_per_shard, 
                    count_per_shard * sizeof(float), cudaMemcpyDeviceToDevice, data_stream_);
    std::swap(tensor_, temp_tensor_);
    shape_ = local_shape;
    size_ = count_per_shard;
    recordCommDone();
}

void DTensor::allGather_async() {
    size_t count_per_rank = size_;
    recordComputeDone();
    waitForCompute();
    std::vector<int> global_shape = layout_.get_global_shape();
    ensureTempTensor(global_shape);
    pg_->all_gather_async(tensor_.data<float>(), temp_tensor_.data<float>(), count_per_rank, OwnTensor::Dtype::Float32, false);
    std::swap(tensor_, temp_tensor_);
    shape_ = global_shape;
    size_ = tensor_.numel();
    recordCommDone();
}

// ============================================================================
// Phase 4: Fused Operations (reduce memory bandwidth)
// ============================================================================

void DTensor::add_bias(const DTensor& bias) {
    recordComputeDone();
    waitForComm();
    
    launch_bias_kernel(tensor_.data<float>(), bias.tensor_.data<float>(),
                       size_, bias.size_, compute_stream_);
    
    recordComputeDone();
}

DTensor DTensor::matmul_bias_gelu(const DTensor& weights, const DTensor& bias) const {
    // Matmul
    OwnTensor::Tensor result_tensor = TensorOpsBridge::matmul(tensor_, weights.tensor_);
    
    DTensor result(device_mesh_, pg_, result_tensor, layout_);
    
    // Fused bias + GELU (single memory pass instead of two)
    launch_bias_activation_kernel(
        result.tensor_.data<float>(),
        bias.tensor_.data<float>(),
        result.size_,
        bias.size_,
        ActivationType::GELU,
        result.compute_stream_
    );
    
    return result;
}

DTensor DTensor::matmul_bias_relu(const DTensor& weights, const DTensor& bias) const {
    // Matmul
    OwnTensor::Tensor result_tensor = TensorOpsBridge::matmul(tensor_, weights.tensor_);
    
    DTensor result(device_mesh_, pg_, result_tensor, layout_);
    
    // Fused bias + ReLU (single memory pass)
    launch_bias_activation_kernel(
        result.tensor_.data<float>(),
        bias.tensor_.data<float>(),
        result.size_,
        bias.size_,
        ActivationType::RELU,
        result.compute_stream_
    );
    
    return result;
}

// ============================================================================
// Static Factory Functions (PyTorch-style DTensor creation)
// ============================================================================

/**
 * Helper: Convert std::vector<int> to OwnTensor::Shape
 */
static OwnTensor::Shape toShape(const std::vector<int>& dims) {
    OwnTensor::Shape shape;
    shape.dims.assign(dims.begin(), dims.end());
    return shape;
}

/**
 * Helper: Get TensorOptions for the given rank
 */
static OwnTensor::TensorOptions getOpts(int rank) {
    return OwnTensor::TensorOptions()
        .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank))
        .with_dtype(OwnTensor::Dtype::Float32);
}

DTensor DTensor::empty(const std::vector<int>& global_shape, std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg, const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int> local_shape = layout.get_local_shape(rank);
    
    // empty() just allocates memory without initialization - use Tensor constructor directly
    OwnTensor::Tensor local_tensor(toShape(local_shape), getOpts(rank));
    
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::zeros(const std::vector<int>& global_shape, std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg, const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int> local_shape = layout.get_local_shape(rank);
    
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::zeros(toShape(local_shape), getOpts(rank));
    
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::ones(const std::vector<int>& global_shape, std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg, const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int> local_shape = layout.get_local_shape(rank);
    
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::ones(toShape(local_shape), getOpts(rank));
    
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::full(const std::vector<int>& global_shape, float value, std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg, const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int> local_shape = layout.get_local_shape(rank);
    
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::full(toShape(local_shape), getOpts(rank), value);
    
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::rand(const std::vector<int>& global_shape, std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg, const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int> local_shape = layout.get_local_shape(rank);
    
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::rand(toShape(local_shape), getOpts(rank));
    
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::randn(const std::vector<int>& global_shape, std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg, const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int> local_shape = layout.get_local_shape(rank);
    
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::randn(toShape(local_shape), getOpts(rank));
    
    return DTensor(mesh, pg, local_tensor, layout);
}

DTensor DTensor::randint(int64_t low, int64_t high, const std::vector<int>& global_shape, std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg,  const Layout& layout) {
    int rank = pg->get_rank();
    std::vector<int> local_shape = layout.get_local_shape(rank);
    
    // randint: Generate uniform [0,1) then scale to [low, high) integers
    // Use rand() which exists in OwnTensor, then scale on CPU and upload
    OwnTensor::Tensor local_tensor = OwnTensor::Tensor::rand(toShape(local_shape), getOpts(rank));
    
    // Download, scale to integers, upload
    size_t numel = 1;
    for (int d : local_shape) numel *= d;
    
    std::vector<float> data(numel);
    cudaMemcpy(data.data(), local_tensor.data<float>(), numel * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < numel; ++i) {
        data[i] = static_cast<float>(static_cast<int64_t>(data[i] * (high - low) + low));
    }
    
    cudaMemcpy(local_tensor.data<float>(), data.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
    
    return DTensor(mesh, pg, local_tensor, layout);
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
    int world_size = pg->get_worldsize();
    
    std::vector<int> global_shape = layout.get_global_shape();
    std::vector<int> local_shape = layout.get_local_shape(rank);
    
    size_t global_size = 1;
    for (int d : global_shape) global_size *= d;
    
    size_t local_size = 1;
    for (int d : local_shape) local_size *= d;
    
    // Allocate local tensor
    OwnTensor::Tensor local_tensor(toShape(local_shape), getOpts(rank));
    
    // Case 1: Replicated layout - Broadcast full tensor to all ranks
    if (layout.is_replicated()) {
        // All ranks need full-size tensor for replicated layout
        OwnTensor::Tensor full_tensor(toShape(global_shape), getOpts(rank));
        
        // Copy from root's global_tensor to full_tensor (or allocate on non-root)
        if (rank == root) {
            cudaMemcpy(full_tensor.data<float>(), global_tensor.data<float>(),
                       global_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        
        // Broadcast to all ranks
        pg->broadcast(full_tensor.data<float>(), full_tensor.data<float>(),
                             global_size, OwnTensor::Dtype::Float32, root, true);
        
        return DTensor(mesh, pg, full_tensor, layout);
    }
    
    // Case 2: Sharded layout
    int shard_dim = layout.get_shard_dim();
    
    if (shard_dim == 0) {
        // Row-sharded: Use scatter directly (contiguous chunks)
        pg->scatter(
            (rank == root) ? global_tensor.data<float>() : nullptr,
            local_tensor.data<float>(),
            local_size,
            OwnTensor::Dtype::Float32,
            root,
            true
        );
        
    } else if (shard_dim == 1) {
        // Column-sharded: Broadcast full tensor, then extract local shard
        OwnTensor::Tensor full_tensor(toShape(global_shape), getOpts(rank));
        
        if (rank == root) {
            cudaMemcpy(full_tensor.data<float>(), global_tensor.data<float>(),
                       global_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        
        pg->broadcast(full_tensor.data<float>(), full_tensor.data<float>(),
                             global_size, OwnTensor::Dtype::Float32, root, true);
        
        // Extract column shard (non-contiguous 2D slice)
        int rows = global_shape[0];
        int global_cols = global_shape[1];
        int local_cols = local_shape[1];
        int col_offset = rank * local_cols;
        
        // Use a synchronous copy since we don't have stream pool here
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
