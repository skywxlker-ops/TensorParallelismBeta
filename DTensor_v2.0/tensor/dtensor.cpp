#include <nccl.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <numeric> 
#include <stdexcept> 
#include <sstream>
#include "tensor/dtensor.h"
#include "TensorLib.h"


CachingAllocator gAllocator;
// using namespace OwnTensor;

DTensor::DTensor(DeviceMesh device_mesh, std::shared_ptr<ProcessGroup> pg, Layout layout)
    : rank_(pg->getRank()),
      world_size_(pg->getWorldSize()),// worldsize is no. of GPUs in a group.
      device_mesh_(device_mesh),
      pg_(pg),
      stream_(pg->getStream()),
      layout_(layout), 
      size_(0),
      shape_(0),
      tensor_()
    //   data_block_(nullptr),
      { 
        shape_ = layout.get_global_shape();
        // if(rank_ == 0 ) {
        //     std::cout <<"Dtensor Constructor: \n";
        //     for(auto i: shape_) { std::cout << i << " ";}
        //     std::cout<<"\n";
        // }
        OwnTensor::TensorOptions opts;
        opts.dtype = OwnTensor::Dtype::Float32;
        opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_);
        OwnTensor::Shape shape{shape_};
        tensor_ = OwnTensor::Tensor(shape, opts);
        size_ = 1;
        for (int d : layout_.get_global_shape()) size_ *= d;        
      }

// DTensor::DTensor(std::shared_ptr<DeviceMesh> device_mesh,
//                  std::shared_ptr<ProcessGroup> pg,
//                  const OwnTensor::Tensor& local_tensor,
//                  const Layout& layout)
//     : rank_(pg->getRank()),
//       world_size_(pg->getWorldSize()),
//       device_mesh_(device_mesh),
//       pg_(pg),
//       stream_(pg->getStream()),
//       layout_(layout),
//       tensor_(local_tensor)//,           
//       //temp_tensor_(local_tensor)      
// {
//     cudaSetDevice(rank_);
    

//     shape_ = layout_.get_local_shape(rank_); 
//     size_ = 1;
//     for (int d : shape_) size_ *= d;

//     data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
    
// }

DTensor::~DTensor() {
  
    cudaStreamSynchronize(stream_); 
    // if (data_block_) gAllocator.freeMemory(data_block_);
    // tensor_.release();
}

void DTensor::setData(const std::vector<float>& host_data) {

    if (host_data.size() != (size_t)size_) {
        std::ostringstream oss;
        oss << "DTensor::setData: host_data size (" << host_data.size() 
            << ") does not match calculated local shard size (" << size_ << ")."
            << " Rank: " << rank_ << ", " << layout_.describe(rank_);
        throw std::runtime_error(oss.str());
    }

    // OwnTensor::Shape shape_obj;
    // shape_obj.dims.assign(local_shape.begin(), local_shape.end());

    // OwnTensor::TensorOptions opts;
    // opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
    //            .with_dtype(OwnTensor::Dtype::Float32);

    // tensor_ = OwnTensor::Tensor(shape_obj, opts);
    tensor_.set_data(host_data);

    // temp_tensor_ = OwnTensor::Tensor(shape_obj, opts);

    // if (data_block_) gAllocator.freeMemory(data_block_);
    // if (temp_block_) gAllocator.freeMemory(temp_block_);
    
    // data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
    // temp_block_ = gAllocator.allocateMemory(layout_.global_numel() * sizeof(float), stream_);
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
    
    pg_->broadcast<float>(tensor_.data<float>(), total_numel, root, ncclFloat)->wait();
    
    layout_ = Layout(device_mesh_, global_shape);
    shape_ = global_shape;
    size_ = total_numel;
}


// void DTensor::shard(int dim, int root, DTensor &parent_tensor) {
//     std::vector<int> global_shape = parent_tensor.layout_.get_global_shape();
    
//     if (dim < 0 || dim >= (int)global_shape.size()) {
//         std::ostringstream oss;
//         oss << "DTensor::shard: Invalid shard dimension " << dim 
//             << " for tensor with " << global_shape.size() << " dimensions";
//         throw std::runtime_error(oss.str());
//     }
    
//     std::vector<int> local_shape = parent_tensor.layout_.get_local_shape(rank_);
//     size_t shard_numel = 1;
//     for (int d : local_shape) shard_numel *= d;
    
//     pg_->scatter<float>(
//         parent_tensor.tensor_.data<float>(),
//         shard_numel,
//         root,
//         ncclFloat
//     )->wait();
    
    

//     // OwnTensor::Shape local_shape_obj;
//     // local_shape_obj.dims.assign(local_shape.begin(), local_shape.end());
//     // OwnTensor::Tensor shard_tensor(local_shape_obj,
//     //                                OwnTensor::TensorOptions()
//     //                                    .with_device(tensor_.device())
//     //                                    .with_dtype(tensor_.dtype()));

//     cudaMemcpyAsync(
//         tensor_.data<float>(),
//         parent_tensor.tensor_.data<float>() + rank_ * shard_numel,
//         shard_numel * sizeof(float),
//         cudaMemcpyDeviceToDevice,
//         stream_
//     );
    
//     cudaStreamSynchronize(stream_);
    
//     ~parent_tensor();
//     // tensor_ = std::move(shard_tensor);
//     // layout_ = sharded_layout;
//     // shape_ = local_shape;
//     // size_ = shard_numel;

//     // shard_tensor.release() 
// }

void DTensor::rotate3D( int dim, bool direction) {
    std::vector<int64_t> n = this->get_layout().get_global_shape();
    // Treat each Z-layer as a 2D matrix and rotate it

    float* data = static_cast<float*>(this->tensor_.data());
    if( dim == 0){
    // shape n = {nx, ny, nz}
    for (int64_t x = 0; x < n[0]; ++x) {
        // 1. Transpose Y and Z
        for (int64_t y = 0; y < n[1]; ++y) {
            for (int64_t z = y + 1; z < n[2]; ++z) {
                std::swap(*(data + x*n[1]*n[2] + y*n[2] + z), 
                          *(data + x*n[1]*n[2] + z*n[2] + y));
            }
        }
        // 2. Reverse Z-rows (Contiguous)
        for (int64_t y = 0; y < n[1]; ++y) {
            float* zStart = data + x*n[1]*n[2] + y*n[2];
            std::reverse(zStart, zStart + n[2]);
        }
    }
    }

    if( dim == 1){
    for (int64_t y = 0; y < n[1]; ++y) {
        // 1. Transpose X and Z
        for (int64_t x = 0; x < n[0]; ++x) {
            for (int64_t z = x + 1; z < n[2]; ++z) {
                std::swap(*(data + x*n[1]*n[2] + y*n[2] + z), 
                          *(data + z*n[1]*n[2] + y*n[2] + x));
            }
        }

        for (int64_t x = 0; x < n[0]; ++x) {
            float* zStart = data + x*n[1]*n[2] + y*n[2];
            std::reverse(zStart, zStart + n[2]);
        }
    }
    }

    if( dim == 2){
    for (int64_t z = 0; z < n[2]; ++z) {
        // 1. Transpose X and Y
        for (int64_t x = 0; x < n[0]; ++x) {
            for (int64_t y = x + 1; y < n[1]; ++y) {
                std::swap(*(data + x*n[1]*n[2] + y*n[2] + z), 
                          *(data + y*n[1]*n[2] + x*n[2] + z));
            }
        }
        // 2. Manual Reverse of X-coordinates (Non-contiguous)
        for (int64_t y = 0; y < n[1]; ++y) {
            for (int64_t x = 0; x < n[0] / 2; ++x) {
                std::swap(*(data + x*n[1]*n[2] + y*n[2] + z), 
                          *(data + (n[0]-1-x)*n[1]*n[2] + y*n[2] + z));
            }
        }
    }
    }
}

    // for (int64_t y = 0; y < n[1]; ++y) {
    
    //     // 1. Transpose x and y for this specific z
    //     for (int64_t x = 0; x < n[0]; ++x) {
    //         for (int64_t z = x + 1; z < n[1]; ++z) {
    //             // Swap element at (x, y, z) with (y, x, z)
    //             float* val1 = data + (x * n[1] * n[2]) + (y * n[2]) + z;
    //             float* val2 = data + (z * n[1] * n[2]) + (y * n[2]) + x;
    //             std::swap(*val1, *val2);
    //         }
    //     }
    
    //     // 2. Reverse (to complete the rotation)
    //     // Note: Reversing a "row" in this layout is difficult with std::reverse 
    //     // because the x-elements for a fixed z are not contiguous in memory.
    //     for (int64_t z = 0; z < n[2]; ++z) {
    //         for (int64_t x = 0; x < n[0] / 2; ++x) {
    //             float* left  = data + (x * n[1] * n[2]) + (y * n[2]) + z;
    //             float* right = data + ((n[0] - 1 - x) * n[1] * n[2]) + (y * n[2]) + z;
    //             std::swap(*left, *right);
    //         }
    //     }
    // }    

    // for (int64_t y = 0; y < n[1]; ++y) {
    //     // 1. Transpose the X-Z plane for this Y-slice
    //     for (int64_t x = 0; x < n[0]; ++x) {
    //         for (int64_t z = x + 1; z < n[2]; ++z) {
    //             // Keep 'y' fixed, swap x and z
    //             std::swap(*(data + x*n[0]*n[1] + y*n[0] + z), *(data + z*n[0]*n[1] + y*n[0] + x));
    //         }
    //     }
    //     // 2. Reverse each "row" in the X-Z plane (reverse Z for each X)
    //     for (int64_t x = 0; x < n[0]; ++x) {
    //         for (int64_t z = 0; z < n[2] / 2; ++z) {
    //             std::swap(*(data + x*n[0]*n[1] + y*n[0] + z), *(data + x*n[0]*n[1] + y*n[0] + n[2] - 1 - z));
    //         }
    //     }
    // }

void DTensor::shard(int dim, int root, DTensor &parent_tensor) {
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
                pg_->scatter<float>(
                parent_tensor.tensor_.data<float>() + m * n * global_shape[2] ,
                shard_numel / ( global_shape[0] * global_shape[1] ),
                root,
                ncclFloat
                )->wait();
                }
            }   
            if(dim == 1){
            pg_->scatter<float>(
            parent_tensor.tensor_.data<float>() + m * global_shape[1] * global_shape[2]  ,
            shard_numel / global_shape[0],
            root,
            ncclFloat
            )->wait();
            }
        }
        if(dim == 0){
        pg_->scatter<float>(
        parent_tensor.tensor_.data<float>() ,
        shard_numel ,
        root,
        ncclFloat
        )->wait();
        }
    for (int64_t m = 0; m < global_shape[0]; m++){
        for(int64_t n = 0; n < global_shape[1]; n++){
            if(dim == 2){ 
            cudaMemcpyAsync(
            tensor_.data<float>() + m * n * (shard_numel/( global_shape[0] * global_shape[1] )),
            parent_tensor.tensor_.data<float>() + m * global_shape[1] * global_shape[2]  +  n * global_shape[2]  + rank_ * (shard_numel/ (world_size_ << 1)),
            shard_numel * sizeof(float)/( global_shape[0] * global_shape[1] ),
            cudaMemcpyDeviceToDevice,
            stream_
            );
            cudaStreamSynchronize(stream_);
            }
        }
        if(dim == 1){ 
        cudaMemcpyAsync(
        tensor_.data<float>() + m * (shard_numel/global_shape[0]),
        parent_tensor.tensor_.data<float>() + m * global_shape[1] * global_shape[2]   + rank_ * (shard_numel/world_size_),
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
    
    
    parent_tensor.tensor_.release();

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
    
    pg_->allGather<float>(
        tensor_.data<float>(),
        shard_numel,
        ncclFloat
    )->wait();

}



void DTensor::sync() {
    pg_->allReduce<float>(tensor_.data<float>(), size_, ncclFloat, ncclSum)->wait();
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
        std:: cout<<"A"<< "["<<A.layout_.get_global_shape()[0]<<", "<<A.layout_.get_global_shape()[1]<<", "<<A.layout_.get_global_shape()[2]<<"] "<<std::endl;
        std:: cout<<"B"<< "["<<B.layout_.get_global_shape()[0]<<", "<<B.layout_.get_global_shape()[1]<<", "<<B.layout_.get_global_shape()[2]<<"] "<<std::endl;
    
        std:: cout<<"shape"<< "["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] "<<std::endl;

        throw std::runtime_error("DTensor shape doesnt match matmul output shape ");
    }
    std:: cout<< "\n Matmul \n";
    std:: cout<< "["<<A.layout_.get_global_shape()[0]<<", "<<A.layout_.get_global_shape()[1]<<", "<<A.layout_.get_global_shape()[2]<<"] "<<std::endl;
    std:: cout<< "["<<B.layout_.get_global_shape()[0]<<", "<<B.layout_.get_global_shape()[1]<<", "<<B.layout_.get_global_shape()[2]<<"] "<<std::endl;
    
    std:: cout<< "["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] "<<std::endl;
    
    tensor_ = TensorOpsBridge::matmul(A.tensor_, B.tensor_);
}



// DTensor DTensor::matmul(const DTensor& other) const {
//     const Layout& a_layout = this->layout_;
//     const Layout& b_layout = other.get_layout();

//     auto a_placement = a_layout.get_placement(0); 
//     auto b_placement = b_layout.get_placement(0);


//     if (a_placement->type() == PlacementType::REPLICATE &&
//         b_placement->type() == PlacementType::SHARD &&
//         static_cast<const Shard*>(b_placement.get())->dim() == 2) {
//         return _column_parallel_matmul(other);
//     }

//     if (a_placement->type() == PlacementType::SHARD &&
//         static_cast<const Shard*>(a_placement.get())->dim() == 2 &&
//         b_placement->type() == PlacementType::SHARD &&
//         static_cast<const Shard*>(b_placement.get())->dim() == 1) {
//          return _row_parallel_matmul(other);
//     }

//     std::ostringstream oss;
//     oss << "DTensor::matmul: This sharding combination is not implemented!\n"
//         << "  Layout A: " << a_layout.describe(rank_) << "\n"
//         << "  Layout B: " << b_layout.describe(rank_);
//     throw std::runtime_error(oss.str());
// }

// DTensor DTensor::_column_parallel_matmul(const DTensor& other) const {
//     OwnTensor::Tensor Y_shard = TensorOpsBridge::matmul(this->tensor_, other.tensor_);
//     std::vector<int64_t> Y_global_shape = {
//         this->layout_.get_global_shape()[0],
//         other.get_layout().get_global_shape()[1]
//     };
//     Layout Y_layout(DTensor::device_mesh_, Y_global_shape, 1);
    
//     return DTensor(DTensor::device_mesh_, pg_, Y_shard, Y_layout);
// }

// DTensor DTensor::_row_parallel_matmul(const DTensor& other) const {
//     OwnTensor::Tensor Y_partial = TensorOpsBridge::matmul(this->tensor_, other.tensor_);
//     std::vector<int64_t> Y_global_shape = {
//         this->layout_.get_global_shape()[0],
//         other.get_layout().get_global_shape()[1]
//     };
//     Layout Y_layout(device_mesh_, Y_global_shape, 2);

//     return  DTensor (device_mesh_, pg_, Y_partial, Y_layout);
// }

// DTensor DTensor::_ring(const DTensor& other) const {



//  DTensor DTensor::reshape(const std::vector<int64_t>& new_global_shape) const {

//     if (layout_.is_sharded()) {
//         throw std::runtime_error("DTensor::reshape: Reshaping sharded tensors not yet implemented");
//     }


//     int new_local_size = 1;
//     for (int d : new_global_shape) new_local_size *= d;
//     if (new_local_size != size_) {
//         throw std::runtime_error("DTensor::reshape: local element count mismatch.");
//     }

//     OwnTensor::Shape shape_obj;
//     shape_obj.dims.assign(new_global_shape.begin(), new_global_shape.end());



//     Layout new_layout(device_mesh_, new_global_shape);

//     return DTensor(device_mesh_, pg_,  new_layout);
// }


// Checkpointing (N-D Safe)

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
    
//     std::vector<int64_t> loaded_shape(ndim);
//     file.read(reinterpret_cast<char*>(loaded_shape.data()), ndim * sizeof(int));
//     char dtype_buf[32];
//     file.read(dtype_buf, sizeof(dtype_buf));
//     dtype_ = std::string(dtype_buf);
    
//     int loaded_size = 1;
//     for (int d : loaded_shape) loaded_size *= d;

//     std::vector<float> host_data(loaded_size);
//     file.read(reinterpret_cast<char*>(host_data.data()), loaded_size * sizeof(float));
//     file.close();

//     Layout loaded_layout(device_mesh_, loaded_shape);
    
//     setData(host_data);

//     std::cout << "[Rank " << rank_ << "] Checkpoint loaded: " << path
//               << " (" << ndim << "D, " << size_ << " elements)\n";
// }

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
    tensor_.to_cpu().display();
}

void DTensor::rand() {
    OwnTensor::TensorOptions opts;
        opts.dtype = OwnTensor::Dtype::Float32;
        opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_);
    tensor_  = OwnTensor::Tensor::rand({shape_}, opts);
}