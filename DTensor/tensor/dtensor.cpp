#include <nccl.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <numeric> 
#include <stdexcept> 
#include <sstream>
#include "dtensor.h"
#include "TensorLib.h"
#include "autograd/ops_template.h"
#include "mlp/layers.h"
#include "device/DeviceCore.h"  // For OwnTensor::cuda::setCurrentStream

// #include <nvtx3/nvToolsExt.h>
#include "process_group/fused_transpose_kernel.cuh"



// =============================================================================
// AllReduceSumBackward: Autograd node for distributed gradient sync
// Forward: Y = all_reduce_sum(X)  (sum partial results across GPUs)
// Backward: dX = all_reduce_sum(dY) (sum gradient contributions across GPUs)
// =============================================================================
class AllReduceSumBackward : public OwnTensor::Node {
private:
    std::shared_ptr<ProcessGroupNCCL> pg_;
    size_t numel_;
    OwnTensor::Dtype dtype_;
    
public:
    AllReduceSumBackward(std::shared_ptr<ProcessGroupNCCL> pg, size_t numel, OwnTensor::Dtype dtype)
        : Node(1), pg_(pg), numel_(numel), dtype_(dtype) {}
    
    const char* name() const override { return "AllReduceSumBackward"; }
    
    std::vector<OwnTensor::Tensor> apply(std::vector<OwnTensor::Tensor>&& grads) override {
        if (grads.empty()) return {};
        
        OwnTensor::Tensor grad = grads[0];
        
        // All-reduce gradient: sum contributions from all GPUs
        // This is critical for correct gradient flow through tensor parallel sync points
        pg_->all_reduce_async(
            grad.data(), 
            const_cast<void*>(grad.data()), 
            grad.numel(), 
            grad.dtype(), 
            sum, 
            false
        )->wait();
        
        return {grad};
    }
    
    // CRITICAL: Clear pg_ shared_ptr to break reference cycle and allow cleanup
    void release_saved_variables() override {
        pg_.reset();
    }
};


// Default constructor for member initialization
DTensor::DTensor()
    : rank_(0),
      world_size_(1),
      device_mesh_(nullptr),
      pg_(nullptr),
      stream_(nullptr),
      layout_(),
      size_(0),
      shape_(),
      tensor_()
{

}

DTensor::DTensor(const DeviceMesh& device_mesh, std::shared_ptr<ProcessGroupNCCL> pg, Layout layout, std::string name, float sd, int seed)
    : rank_(pg->get_rank()),
      world_size_(pg->get_worldsize()),// worldsize is no. of GPUs in a group.
      device_mesh_(&device_mesh),
      pg_(pg),
      stream_(pg->getStream()),
      layout_(layout), 
      size_(0),
      shape_(0),
      tensor_()
      { 
        shape_ = layout.get_global_shape();
        
        // Calculate local GPU device based on global MPI rank, not process group rank
        int global_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
        
        // Auto-detect GPUs per node, can be overridden by NO_GPUS_PER_NODE env var
        int gpus_per_node = 1;
        cudaGetDeviceCount(&gpus_per_node);
        const char* env = std::getenv("NO_GPUS_PER_NODE");
        if (env) {
            int parsed = std::atoi(env);
            if (parsed > 0) gpus_per_node = parsed;
        }
        int local_gpu = global_rank % gpus_per_node;
        
        // Ensure we're on the correct device before creating tensor
        cudaSetDevice(local_gpu);
        
        #ifdef WITH_CUDA
        OwnTensor::cuda::setCurrentStream(stream_);
        #endif
        
        OwnTensor::TensorOptions opts;
        opts.dtype = OwnTensor::Dtype::Float32;
        opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, local_gpu);
        OwnTensor::Shape shape{shape_};

        
        tensor_ = OwnTensor::Tensor::randn<float>( shape, opts, seed , sd) ;
        
        size_ = tensor_.numel();
        
        name_ = name;

    }




DTensor::~DTensor() {
    // Guard against invalid stream - can happen if ProcessGroup is destroyed first
    if (stream_ != nullptr) {
        cudaError_t err = cudaStreamSynchronize(stream_);
        // Ignore errors during cleanup - stream may be invalid
        (void)err;
    }
}

void DTensor::setData(const std::vector<float>& host_data) {

    if (host_data.size() != (size_t)size_) {
        std::ostringstream oss;
        oss << "DTensor::setData: host_data size (" << host_data.size() 
            << ") does not match calculated local shard size (" << size_ << ")."
            << " Rank: " << rank_ << ", " << layout_.describe(rank_);
        throw std::runtime_error(oss.str());
    }

    tensor_.set_data(host_data);

}
void DTensor::setData(const std::vector<int64_t>& host_data) {

    if (host_data.size() != (size_t)size_) {
        std::ostringstream oss;
        oss << "DTensor::setData: host_data size (" << host_data.size() 
            << ") does not match calculated local shard size (" << size_ << ")."
            << " Rank: " << rank_ << ", " << layout_.describe(rank_);
        throw std::runtime_error(oss.str());
    }

    tensor_.set_data(host_data);

}

std::vector<float> DTensor::getData() const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return host_data;
}


void DTensor::replicate(int root) {
    std::vector<int64_t> global_shape = layout_.get_global_shape();
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
    
    pg_->broadcast_async(tensor_.data<float>(), tensor_.data<float>(), total_numel, OwnTensor::Dtype::Float32, root)->wait();
    
    layout_ = Layout(*device_mesh_, global_shape);
    shape_ = global_shape;
    size_ = total_numel;
}


void DTensor::shard(int dim, int root, DTensor &parent_tensor) {

    // CRITICAL: Sync OwnTensor stream with DTensor stream to prevent race conditions
    #ifdef WITH_CUDA
    OwnTensor::cuda::setCurrentStream(stream_);
    #endif
    std::vector<int64_t> global_shape = parent_tensor.shape_;
    
    if (dim < 0 || dim >= (int)global_shape.size()) {
        std::ostringstream oss;
        oss << "DTensor::shard: Invalid shard dimension " << dim 
            << " for tensor with " << global_shape.size() << " dimensions";
        throw std::runtime_error(oss.str());
    }
    
    std::vector<int64_t> local_shape = parent_tensor.layout_.get_local_shape(rank_);
    size_t shard_numel = 1;
    for (int d : local_shape) shard_numel *= d;


    for (int64_t m = 0; m < global_shape[0]; m++){
        for(int64_t n = 0; n < global_shape[1]; n++){
                if(dim == 2){
                pg_->scatter_async(
                parent_tensor.tensor_.data<float>() + m * global_shape[1] * global_shape[2] +  n * global_shape[2]  ,
                parent_tensor.tensor_.data<float>() + m * global_shape[1] * global_shape[2] +  n * global_shape[2] + rank_ * (shard_numel/ ( global_shape[0] * global_shape[1] )),
                shard_numel / ( global_shape[0] * global_shape[1] ),
                OwnTensor::Dtype::Float32,
                root,
                true
                )->wait();
                }
            }   
            if(dim == 1){
            pg_->scatter_async(
            parent_tensor.tensor_.data<float>() + m * global_shape[1] * global_shape[2]  ,
            parent_tensor.tensor_.data<float>() +  m * global_shape[1] * global_shape[2] + rank_ * ( shard_numel /  global_shape[0] ),
            shard_numel / global_shape[0],
            OwnTensor::Dtype::Float32,
            root,
            true
            )->wait();
            }
        }
        if(dim == 0){
        pg_->scatter_async(
        parent_tensor.tensor_.data<float>() ,
        parent_tensor.tensor_.data<float>() + rank_ * shard_numel,
        shard_numel ,
        OwnTensor::Dtype::Float32,
        root,
        true
        )->wait();
        }
    for (int64_t m = 0; m < global_shape[0]; m++){
        for(int64_t n = 0; n < global_shape[1]; n++){
            if(dim == 2){ 
            cudaMemcpyAsync(
            tensor_.data<float>() + m * global_shape[1] * global_shape[2]/world_size_ +  n * global_shape[2]/world_size_,
            parent_tensor.tensor_.data<float>() + m * global_shape[1] * global_shape[2]  +  n * global_shape[2]  + rank_ * (shard_numel/ ( global_shape[0] * global_shape[1] )),
            shard_numel * sizeof(float)/( global_shape[0] * global_shape[1] ),
            cudaMemcpyDeviceToDevice,
            stream_
            );
            cudaStreamSynchronize(stream_);
            }
        }
        if(dim == 1){ 
        cudaMemcpyAsync(
        tensor_.data<float>() + m * global_shape[1]/world_size_ * global_shape[2],
        parent_tensor.tensor_.data<float>() + m * global_shape[1] * global_shape[2]   + rank_ * (shard_numel/global_shape[0]),
        shard_numel * sizeof(float)/global_shape[0],
        cudaMemcpyDeviceToDevice,
        stream_
        );
        cudaStreamSynchronize(stream_);
        }   
    }
    if(dim == 0){ 
    cudaMemcpyAsync(
        tensor_.data<float>(),
        parent_tensor.tensor_.data<float>() + rank_ * shard_numel,
        shard_numel * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream_
    );
    cudaStreamSynchronize(stream_);
    }
    
}



void DTensor::shard_fused_transpose(int dim, int root, DTensor &parent_tensor) {
    // Shard using custom kernel to extract non-contiguous slices
    // Supports dim 0, 1, 2 for 3D tensors
    // Uses separate send_buffer to avoid race conditions
    
    #ifdef WITH_CUDA
    OwnTensor::cuda::setCurrentStream(stream_);
    #endif
    
    std::vector<int64_t> parent_shape = parent_tensor.shape_;
    std::vector<int64_t> local_shard_shape = parent_tensor.get_layout().get_local_shape(rank_);
    int ndim = parent_shape.size();
    
    if (ndim != 2 && ndim != 3) {
        throw std::runtime_error("shard_fused_transpose: only supports 2D and 3D tensors, got " + std::to_string(ndim) + "D");
    }
    
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("shard_fused_transpose: dim " + std::to_string(dim) + " out of range for " + std::to_string(ndim) + "D tensor");
    }
    
    // Calculate local dimension size and shard numel
    int64_t local_dim_size;
    size_t shard_numel = tensor_.numel();
    
    if (dim == 0) {
        local_dim_size = local_shard_shape[0] / world_size_;
    } else if (dim == 1) {
        local_dim_size = local_shard_shape[1] / world_size_;
    } else { // dim == 2, only valid for 3D
        local_dim_size = local_shard_shape[2] / world_size_;
        shard_numel = local_shard_shape[0] * local_shard_shape[1] * local_dim_size;
    }
    
    std::shared_ptr<Work> work;
    
    if (rank_ == root) {
        // For dim 0, data is already contiguous - just scatter directly
        if (dim == 0) {
            work = pg_->scatter_async(
                parent_tensor.tensor_.data<float>(),
                tensor_.data<float>(),
                shard_numel,
                OwnTensor::Dtype::Float32,
                root,
                true
            );
        } else {
            // For dim 1 and 2, need to reorder using kernel
            OwnTensor::Shape send_shape;
            send_shape.dims = {static_cast<int64_t>(tensor_.numel())};
            OwnTensor::Tensor send_buffer(send_shape, parent_tensor.tensor_.dtype(), parent_tensor.tensor_.device());
            
            // nvtxRangePush("shard_fused_transpose");
            for (int r = 0; r < world_size_; r++) {
                if (dim == 1) {
                    launch_shard_dim1_kernel(
                        parent_tensor.tensor_.data<float>(),
                        send_buffer.data<float>() + r * shard_numel,
                        local_shard_shape,
                        local_dim_size,
                        r,
                        shard_numel,
                        stream_
                    );
                } else { // dim == 2
                    launch_shard_dim2_kernel(
                        parent_tensor.tensor_.data<float>(),
                        send_buffer.data<float>() + r * shard_numel,
                        local_shard_shape,
                        local_dim_size,
                        r,
                        shard_numel,
                        stream_
                    );
                }
            }
            cudaStreamSynchronize(stream_);
            // nvtxRangePop();
            
            // DEBUG: Force device sync before NCCL collective
            cudaDeviceSynchronize();
            
            work = pg_->scatter_async(
                send_buffer.data<float>(),
                tensor_.data<float>(),
                shard_numel,
                OwnTensor::Dtype::Float32,
                root,
                true
            );
        }
    } else {
        // Non-root ranks just receive
        work = pg_->scatter_async(
            nullptr,
            tensor_.data<float>(),
            shard_numel,
            OwnTensor::Dtype::Float32,
            root,
            true
        );
    }
    // Record event on NCCL stream and make our stream wait for it (GPU-side sync)
    work->event_record();
    work->streamWait(stream_);

}



void DTensor::assemble(int dim, int root, DTensor &sharded_tensor) {
    std::vector<int64_t> global_shape = layout_.get_global_shape();
    
    if (dim < 0 || dim >= (int)global_shape.size()) {
        std::ostringstream oss;
        oss << "DTensor::shard: Invalid shard dimension " << dim 
            << " for tensor with " << global_shape.size() << " dimensions";
        throw std::runtime_error(oss.str());
    }
    
    std::vector<int64_t> local_shape = layout_.get_local_shape(rank_);
    size_t shard_numel = 1;
    for (int d : local_shape) shard_numel *= d;


    cudaMemcpyAsync(
        tensor_.data<float>() + rank_ * shard_numel,
        sharded_tensor.tensor_.data<float>(),
        shard_numel * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream_
    );
    
    cudaStreamSynchronize(stream_);
    
    pg_->all_gather_async(
        tensor_.data<float>() + rank_ * shard_numel,
        tensor_.data<float>(),
        shard_numel,
        OwnTensor::Dtype::Float32,
        false
    )->wait();


}

void DTensor::sync() {
    // Blocking sync - enqueue all-reduce and wait for completion
    pg_->all_reduce_async(tensor_.data<float>(), tensor_.data<float>(), size_, OwnTensor::Dtype::Float32, sum, false)->wait();
    has_pending_collective_ = false;
    pending_work_ = nullptr;
}

void DTensor::sync_async() {
    // Async sync - enqueue all-reduce but DON'T wait (deferred wait)
    // Store the work handle so we can wait later when data is actually needed
    pending_work_ = pg_->all_reduce_async(tensor_.data<float>(), tensor_.data<float>(), size_, OwnTensor::Dtype::Float32, sum, false);
    has_pending_collective_ = true;
}



void DTensor::sync_w_autograd(op_t op) {
    // Autograd-aware sync: performs all-reduce and sets up backward graph
    // Forward: Y = all_reduce_sum(X) - sum partial results across GPUs
    // Backward: automatically all-reduces gradient dY before flowing to previous ops
    
    // 1. Forward: Blocking all-reduce
    pg_->all_reduce_async(tensor_.data<float>(), tensor_.data<float>(), size_, OwnTensor::Dtype::Float32, op, false)->wait();
    
    // 2. Set up backward graph if tensor requires grad
    if (tensor_.requires_grad()) {
        // Create backward node that will all-reduce gradient
        auto grad_fn = std::make_shared<AllReduceSumBackward>(pg_, size_, tensor_.dtype());
        
        // Connect to existing graph: get edge to current tensor's grad_fn (if any)
        if (tensor_.grad_fn()) {
            grad_fn->set_next_edge(0, OwnTensor::Edge(tensor_.grad_fn(), 0));
        } else {
            // Tensor is a leaf or has no grad_fn - use get_grad_edge pattern
            grad_fn->set_next_edge(0, OwnTensor::autograd::get_grad_edge(tensor_));
        }
        
        // Set this as the new grad_fn for tensor_
        // This inserts AllReduceSumBackward into the graph
        tensor_.set_grad_fn(grad_fn);
    }
    
    has_pending_collective_ = false;
    pending_work_ = nullptr;
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

// //meant for WQKV matrix DTensor
// void DTensor::qkvsplit( DTensor &q, DTensor &k,DTensor &v){
//     for (int i = 0; i < size_; i++  ){
//         if ( i < size_ /3) q.tensor_.data[i%(size_/3)] = tensor_.data[i];
//         else if( i < 2*size_ /3) q.tensor_.data[i%(size_/3)] = tensor_.data[i];
//         else  q.tensor_->data[i%(size_/3)] = tensor_.data[i];
//     }
// }

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
    
   
    int outer_size = 1;  // Product of dims before 'dim'
    int inner_size = 1;  // Product of dims after 'dim'
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
                    size_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);
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
    
    // Copy unpermuted data back to GPU
    cudaMemcpyAsync(tensor_.data<float>(), unpermuted_data.data(),
                    size_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);
}


#define DEFINE_TENSOR_OP(func, op_name) \
DTensor DTensor::func(const DTensor& other) const { \
    if (!layout_.is_compatible(other.get_layout())) { \
        throw std::runtime_error("Incompatible layouts for operation " #op_name); \
    } \
    OwnTensor::Tensor result = TensorOpsBridge::op_name(tensor_, other.tensor_); \
    return DTensor(device_mesh_, pg_, result, layout_); \
}

// DEFINE_TENSOR_OP(add, add)
// DEFINE_TENSOR_OP(sub, sub)
// DEFINE_TENSOR_OP(mul, mul)
// DEFINE_TENSOR_OP(div, div)


void DTensor::matmul( DTensor& A,  DTensor& B)  {

    if( ( shape_[1] != A.layout_.get_global_shape()[1] ) || ( shape_[2] != B.layout_.get_global_shape()[2] ) ){


        throw std::runtime_error("DTensor shape doesnt match matmul output shape ");
    }
    std:: cout<< "\n Matmul \n";

    
    tensor_ = autograd::matmul(A.tensor_, B.tensor_);
}

void DTensor::Linear(  DTensor& Input,  DTensor& Weights,  DTensor& Bias) {

    tensor_ = mlp_forward::linear(Input.tensor_, Weights.tensor_, Bias.tensor_);

}

void DTensor::linear_w_autograd(DTensor& Input, DTensor& Weights, DTensor& Bias) {

    
    OwnTensor::Tensor out = OwnTensor::autograd::matmul(Input.tensor_, Weights.tensor_);
    tensor_ = OwnTensor::autograd::add(out, Bias.tensor_);
}

void DTensor::linear_w_autograd(DTensor& Input, DTensor& Weights) {
    // No-bias version: just matmul
    tensor_ = OwnTensor::autograd::matmul(Input.tensor_, Weights.tensor_);
}

void DTensor::backward() {

    OwnTensor::Tensor loss = OwnTensor::autograd::mean(tensor_);
    OwnTensor::autograd::backward(loss);
}



void DTensor::printRecursive(const std::vector<float>& data,
                             const std::vector<int64_t>& dims,
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
                    stream_);
    cudaStreamSynchronize(stream_);

    std::cout << "[Rank " << rank_ << " Data] ";
    printRecursive(host_data, shape_, 0, 0);
    std::cout << "\n";
}



void DTensor::display(){

    tensor_.display();
}



