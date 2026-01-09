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
#include "mlp/layers.h"
#include "device/DeviceCore.h"  // For OwnTensor::cuda::setCurrentStream
// #include "reverse.cuh"
#include <nvtx3/nvtx3.hpp>
#include <nvtx3/nvToolsExt.h>
#include "process_group/fused_transpose_kernel.cuh"
#include "process_group/fused_rotate_kernel.cuh"
// #include "process_group/fused_transpose_kernel.cuh"


CachingAllocator gAllocator;
// using namespace OwnTensor;

DTensor::DTensor(const DeviceMesh& device_mesh, std::shared_ptr<ProcessGroupNCCL> pg, Layout layout)
    : rank_(pg->get_rank()),
      world_size_(pg->get_worldsize()),// worldsize is no. of GPUs in a group.
      device_mesh_(device_mesh),
      pg_(pg),
      stream_(pg->getStream()),
      layout_(layout), 
      size_(0),
      shape_(0),
      tensor_(),
      value_()
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
        tensor_ = OwnTensor::Tensor(shape, opts);
        size_ = 1;
        for (int d : layout_.get_global_shape()) size_ *= d;  
        
        value_ = ag::make_tensor(tensor_, "");
      
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
    
    pg_->broadcast_async(tensor_.data<float>(), tensor_.data<float>(), total_numel, OwnTensor::Dtype::Float32, root)->wait();
    
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

// void DTensor::rotate3D( int dim, bool direction) {
//     std::vector<int64_t> n = this->get_layout().get_global_shape();
//     // Treat each Z-layer as a 2D matrix and rotate it

//     float* data = static_cast<float*>(this->tensor_.data());
//     if( dim == 0){
//     // shape n = {nx, ny, nz}
//     layout_.set_global_shape({n[0],n[2],n[1]});
//     for (int64_t x = 0; x < n[0]; ++x) {
//         // 1. Transpose Y and Z
//         for (int64_t y = 0; y < n[1]; ++y) {
//             for (int64_t z = y + 1; z < n[2]; ++z) {
//                 std::swap(*(data + x*n[1]*n[2] + y*n[2] + z), 
//                           *(data + x*n[1]*n[2] + z*n[1] + y));
//             }
//         }
//         // 2. Reverse Z-rows (Contiguous)
//         if(direction == 0){
//             for (int64_t y = 0; y < n[1]; ++y) {
//                 for (int64_t z = 0; z < n[2] / 2; ++z) {
//                     std::swap(*(data + x*n[1]*n[2] + z*n[1] + y), 
//                               *(data + x*n[1]*n[2] + (n[2]-1-z)*n[1] + y));
//                 }     
//             } 
//         }
//         else{
//             for (int64_t z = 0; z < n[2]; ++z) {
//                 float* yStart = data + x*n[1]*n[2] + z*n[1];
//                 std::reverse(yStart, yStart + n[2]);
//             }  
//         }
//     }
//     }

//     if( dim == 1){
//     layout_.set_global_shape({n[2],n[1],n[0]});
//     for (int64_t y = 0; y < n[1]; ++y) {
//         // 1. Transpose X and Z
//         for (int64_t x = 0; x < n[0]; ++x) {
//             for (int64_t z = x + 1; z < n[2]; ++z) {
//                 std::swap(*(data + x*n[1]*n[2] + y*n[2] + z), 
//                           *(data + z*n[1]*n[2] + y*n[2] + x));
//             }
//         }
//         if(direction == 0){
//             for (int64_t x = 0; x < n[0]; ++x) {
//                 for (int64_t z = 0; z < n[2] / 2; ++z) {
//                     std::swap(*(data + z*n[1]*n[0] + y*n[0] + x), 
//                               *(data + (n[2]-1-z)*n[1]*n[0] + y*n[0] + x));
//                 }     
//             } 
//         }
//         else{
//             for (int64_t z = 0; z < n[2]; ++z) {
//                 float* xStart = data + z*n[1]*n[0] + y*n[0];
//                 std::reverse(xStart, xStart + n[2]);
//             }
//         }
//     }
//     }

//     if( dim == 2){
//     layout_.set_global_shape({n[1],n[0],n[2]});
//     for (int64_t z = 0; z < n[2]; ++z) {
//         // 1. Transpose X and Y
//         for (int64_t x = 0; x < n[0]; ++x) {
//             for (int64_t y = x + 1; y < n[1]; ++y) {
//                 std::swap(*(data + x*n[1]*n[2] + y*n[2] + z), 
//                           *(data + y*n[0]*n[2] + x*n[2] + z));
//             }
//         }
//         // 2. Manual Reverse of X-coordinates (Non-contiguous)
//         if(direction == 0){
//             for (int64_t x = 0; x < n[0]; ++x) {
//                 for (int64_t y = 0; y < n[1] / 2; ++y) {
//                     std::swap(*(data + y*n[0]*n[2] + x*n[2] + z), 
//                               *(data + y*n[0]*n[2] + (n[0]-1-x)*n[2] + z));
//                 }
//             }            
//         }
//         else{
//             for (int64_t y = 0; y < n[1]; ++y) {
//                 for (int64_t x = 0; x < n[0] / 2; ++x) {
//                     std::swap(*(data + y*n[0]*n[2] + x*n[2] + z), 
//                               *(data + y*n[0]*n[2] + (n[0]-1-x)*n[2] + z));
//                 }
//             }
//         }
//     }
//     }
// }

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

// void DTensor::shard(int dim, int root, DTensor &parent_tensor) {
//     // CRITICAL: Sync OwnTensor stream with DTensor stream to prevent race conditions
//     #ifdef WITH_CUDA
//     OwnTensor::cuda::setCurrentStream(stream_);
//     #endif
//     std::vector<int64_t> global_shape = parent_tensor.shape_;
    
//     if (dim < 0 || dim >= (int)global_shape.size()) {
//         std::ostringstream oss;
//         oss << "DTensor::shard: Invalid shard dimension " << dim 
//             << " for tensor with " << global_shape.size() << " dimensions";
//         throw std::runtime_error(oss.str());
//     }
    
//     std::vector<int64_t> local_shape = parent_tensor.layout_.get_local_shape(rank_);
//     size_t shard_numel = 1;
//     for (int d : local_shape) shard_numel *= d;

//      for (int64_t m = 0; m < global_shape[0]; m++){
//         for(int64_t n = 0; n < global_shape[1]; n++){
//             if(dim == 2){ 
//             cudaMemcpyAsync(
//             tensor_.data<float>() + m * global_shape[1] * global_shape[2]/world_size_ +  n * global_shape[2]/world_size_,
//             parent_tensor.tensor_.data<float>() + m * global_shape[1] * global_shape[2]  +  n * global_shape[2]  + rank_ * (shard_numel/ ( global_shape[0] * global_shape[1] )),
//             shard_numel * sizeof(float)/( global_shape[0] * global_shape[1] ),
//             cudaMemcpyDeviceToDevice,
//             stream_
//             );
//             cudaStreamSynchronize(stream_);
//             }
//         }
//         if(dim == 1){ 
//         cudaMemcpyAsync(
//         tensor_.data<float>() + m * global_shape[1]/world_size_ * global_shape[2],
//         parent_tensor.tensor_.data<float>() + m * global_shape[1] * global_shape[2]   + rank_ * (shard_numel/global_shape[0]),
//         shard_numel * sizeof(float)/global_shape[0],
//         cudaMemcpyDeviceToDevice,
//         stream_
//         );
//         cudaStreamSynchronize(stream_);
//         }   
//     }
//     if(dim == 0){ 
//     cudaMemcpyAsync(
//         tensor_.data<float>(),
//         parent_tensor.tensor_.data<float>() + rank_ * shard_numel,
//         shard_numel * sizeof(float),
//         cudaMemcpyDeviceToDevice,
//         stream_
//     );
//     cudaStreamSynchronize(stream_);
//     }

//     for (int64_t m = 0; m < global_shape[0]; m++){
//         for(int64_t n = 0; n < global_shape[1]; n++){
//                 if(dim == 2){
//                 pg_->scatter_async(
//                 tensor_.data<float>() + m * global_shape[1] * global_shape[2]/world_size_ +  n * global_shape[2]/world_size_  ,
//                 tensor_.data<float>() + m * global_shape[1] * global_shape[2]/world_size_ +  n * global_shape[2]/world_size_ * (shard_numel/ ( global_shape[0] * global_shape[1] )),
//                 shard_numel / ( global_shape[0] * global_shape[1] ),
//                 OwnTensor::Dtype::Float32,
//                 root,
//                 true
//                 )->wait();
//                 }
//             }   
//             if(dim == 1){
//             pg_->scatter_async(
//             tensor_.data<float>() + m * global_shape[1]/world_size_ * global_shape[2] ,
//             tensor_.data<float>() + m * global_shape[1]/world_size_ * global_shape[2] + rank_ * ( shard_numel /  global_shape[0] ),
//             shard_numel / global_shape[0],
//             OwnTensor::Dtype::Float32,
//             root,
//             true
//             )->wait();
//             }
//         }
//         if(dim == 0){
//         pg_->scatter_async(
//         tensor_.data<float>() ,
//         tensor_.data<float>() + rank_ * shard_numel,
//         shard_numel ,
//         OwnTensor::Dtype::Float32,
//         root,
//         true
//         )->wait();
//         }
   
    
//     // Note: Don't release parent tensor here - same issue as shard_default
//     // parent_tensor.tensor_.release();

// }

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
    
    
    // parent_tensor.tensor_.release();

}


// void launch_rotate3D_kernel(float* d_src, float* d_dst, int nx, int ny, int nz, int dny, int dnz, int dim, bool direction, cudaStream_t stream);

// void DTensor::rotate3D(int dim, bool direction) {
//     // 1. Get dimensions
//     std::vector<int64_t> shape = this->get_layout().get_global_shape();
//     int64_t nx = shape[0], ny = shape[1], nz = shape[2];
//     int64_t total_elements = (int64_t)nx * ny * nz;
//     std::cout<<"\n"<<nx<<" "<<ny<<" "<<nz<<"\n";

//     // 2. Determine new dimensions for the destination
//     int dnx = nx, dny = ny, dnz = nz;
    
//     if (dim == 0)      { std::swap(dny, dnz); }
//     else if (dim == 1) { std::swap(dnx, dnz); }
//     else               { std::swap(dnx, dny); }

//     std::cout<<"\n"<<dnx<<" "<<dny<<" "<<dnz<<"\n";
    
//     // 3. Allocate a temporary GPU buffer for the result
//     float* d_src = static_cast<float*>(this->tensor_.data());
//     float* d_dst;
//     cudaMalloc(&d_dst, total_elements * sizeof(float));

//     // 4. Configure Kernel Launch (3D block and grid)
//     dim3 threads(8, 8, 8); 
//     dim3 blocks((nx + threads.x - 1) / threads.x,
//                 (ny + threads.y - 1) / threads.y,
//                 (nz + threads.z - 1) / threads.z);

//     // 5. Launch Kernel on GPU
//     launch_rotate3D_kernel(d_src, d_dst, nx, ny, nz, dny, dnz, dim, direction, stream_);
                                            
//     cudaMemcpyAsync(d_src, d_dst, total_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
//     cudaStreamSynchronize(stream_);
//     cudaFree(d_dst);


//     std::vector<int64_t> new_shape_vec;
//     if (dim == 0)      new_shape_vec = {nx, nz, ny};
//     else if (dim == 1) new_shape_vec = {nz, ny, nx};
//     else               new_shape_vec = {ny, nx, nz};

//     std::cout<<"\n"<<new_shape_vec[0]<<" "<<new_shape_vec[1]<<" "<<new_shape_vec[2]<<"\n";

//     this->shape_ = new_shape_vec;

//     std::cout<<"\n"<<shape_[0]<<" "<<shape_[1]<<" "<<shape_[2]<<"\n";

//     this->layout_.set_global_shape(new_shape_vec);

//     std::cout<<"\n"<<layout_.get_global_shape()[0]<<" "<<layout_.get_global_shape()[1]<<" "<<layout_.get_global_shape()[2]<<"\n";

//     this->tensor_.reshape(OwnTensor::Shape {new_shape_vec});

//     // if (dim == 0)     std::swap(tensor_.shape().dims[1], tensor_.shape().dims[2]);
//     // else if (dim == 1) std::swap(tensor_.shape().dims[0], tensor_.shape().dims[2]);
//     // else               std::swap(tensor_.shape().dims[1], tensor_.shape().dims[0]);

//     std::cout<<"\n"<<tensor_.shape().dims[0]<<" "<<tensor_.shape().dims[1]<<" "<<tensor_.shape().dims[2]<<"\n";

// }

void DTensor::shard_transpose(int dim, int root, DTensor &parent_tensor){
    #ifdef WITH_CUDA
    OwnTensor::cuda::setCurrentStream(stream_);
    #endif
    
    std::vector<int64_t> global_shape = parent_tensor.shape_;
    
    // Calculate shard shape
    // std::vector<int64_t> shard_shape = global_shape;
    // shard_shape[dim] /= world_size_;
    // shape_ = shard_shape;
    
    size_t shard_numel = 1;
    for (int64_t d : shape_) shard_numel *= d;

    if (dim == 0) {
        // Direct scatter - dim 0 is already contiguous
        // OwnTensor::Shape new_shape;
        // new_shape.dims = shape_;
        // tensor_ = OwnTensor::Tensor(new_shape, parent_tensor.tensor_.dtype(), parent_tensor.tensor_.device());
        
        pg_->scatter_async(
            parent_tensor.tensor_.data<float>(),
            tensor_.data<float>(),
            shard_numel,
            OwnTensor::Dtype::Float32,
            root,
            true
        )->wait();
    }
    else if (dim == 2) {
        // Transpose parent_tensor: [X, Y, Z] -> [Z, Y, X]
        // OwnTensor::Tensor transposed
        parent_tensor.tensor_ = parent_tensor.tensor_.transpose(0, 2).contiguous();
        // OwnTensor::Tensor transposed_contig = transposed.contiguous();
        
        // Intermediate shard shape after transpose: [Z/world_size, Y, X]
        int64_t X = global_shape[0], Y = global_shape[1], Z = global_shape[2];
        int64_t Z_local = Z / world_size_;
        
        // Allocate intermediate buffer for scattered data
        // OwnTensor::Shape intermediate_shape;
        // intermediate_shape.dims = 
        tensor_.reshape({{Z_local, Y, X}});
        // OwnTensor::Tensor scattered_transposed(intermediate_shape, parent_tensor.tensor_.dtype(), parent_tensor.tensor_.device());
        
        // Single contiguous scatter on the transposed data
        pg_->scatter_async(
            parent_tensor.tensor_.data<float>(),
            tensor_.data<float>(),
            shard_numel,
            OwnTensor::Dtype::Float32,
            root,
            true
        )->wait();
        
        // Transpose back: [Z/world_size, Y, X] -> [X, Y, Z/world_size]
        tensor_ = tensor_.transpose(0, 2).contiguous();
         
    }
    else if (dim == 1) {
        // Transpose parent_tensor: [X, Y, Z] -> [Y, X, Z]
        parent_tensor.tensor_ = parent_tensor.tensor_.transpose(0, 1).contiguous();
        
        
        int64_t X = global_shape[0], Y = global_shape[1], Z = global_shape[2];
        int64_t Y_local = Y / world_size_;
        
        // OwnTensor::Shape intermediate_shape;
        // intermediate_shape.dims = 
        tensor_.reshape({{Y_local, X, Z}});
        // OwnTensor::Tensor scattered_transposed(intermediate_shape, parent_tensor.tensor_.dtype(), parent_tensor.tensor_.device());
        
        pg_->scatter_async(
            parent_tensor.tensor_.data<float>(),
            tensor_.data<float>(),
            shard_numel,
            OwnTensor::Dtype::Float32,
            root,
            true
        )->wait();
        
        // Transpose back: [Y/world_size, X, Z] -> [X, Y/world_size, Z]
        tensor_ = tensor_.transpose(0, 1).contiguous();
        
    }
    
    setShape(tensor_.shape().dims);
}

// void DTensor::shard_transpose_fused(int dim, int root, DTensor &parent_tensor){
//     #ifdef WITH_CUDA
//     OwnTensor::cuda::setCurrentStream(stream_);
//     #endif
    
//     std::vector<int64_t> global_shape = parent_tensor.shape_;
    
//     size_t shard_numel = 1;
//     for (int64_t d : shape_) shard_numel *= d;
    
//     int64_t X = global_shape[0], Y = global_shape[1], Z = global_shape[2];
//     int64_t total_elements = X * Y * Z;

//     if (dim == 0) {
//         // Direct scatter - dim 0 is already contiguous
//         pg_->scatter_async(
//             parent_tensor.tensor_.data<float>(),
//             tensor_.data<float>(),
//             shard_numel,
//             OwnTensor::Dtype::Float32,
//             root,
//             true
//         )->wait();
//     }
//     else if (dim == 2) {
//         // Fused transpose [X, Y, Z] -> [Z, Y, X] and make contiguous
//         int64_t Z_local = Z / world_size_;
        
//         // Source strides (contiguous [X, Y, Z])
//         int64_t s0 = Y * Z;
//         int64_t s1 = Z;
//         int64_t s2 = 1;
        
//         // Allocate transposed contiguous buffer
//         OwnTensor::Shape transposed_shape;
//         transposed_shape.dims = {Z, Y, X};
//         OwnTensor::Tensor transposed_buf(transposed_shape, parent_tensor.tensor_.dtype(), parent_tensor.tensor_.device());
        
//         // Launch fused transpose kernel
//         launch_fused_transpose_contiguous_kernel(
//             parent_tensor.tensor_.data<float>(),
//             transposed_buf.data<float>(),
//             X, Y, Z,
//             s0, s1, s2,
//             0, 2,  // transpose dim 0 and 2
//             total_elements,
//             stream_
//         );
        
//         // Reshape tensor_ for receiving transposed shard
//         tensor_.reshape({{Z_local, Y, X}});
        
//         // Scatter from transposed buffer
//         pg_->scatter_async(
//             transposed_buf.data<float>(),
//             tensor_.data<float>(),
//             shard_numel,
//             OwnTensor::Dtype::Float32,
//             root,
//             true
//         )->wait();
        
//         // Fused transpose back [Z_local, Y, X] -> [X, Y, Z_local]
//         int64_t rs0 = Y * X;
//         int64_t rs1 = X;
//         int64_t rs2 = 1;
        
//         OwnTensor::Shape result_shape;
//         result_shape.dims = shape_;
//         OwnTensor::Tensor result_buf(result_shape, tensor_.dtype(), tensor_.device());
        
//         launch_fused_transpose_contiguous_kernel(
//             tensor_.data<float>(),
//             result_buf.data<float>(),
//             Z_local, Y, X,
//             rs0, rs1, rs2,
//             0, 2,
//             shard_numel,
//             stream_
//         );
        
//         tensor_ = result_buf;
//     }
//     else if (dim == 1) {
//         // Fused transpose [X, Y, Z] -> [Y, X, Z]
//         int64_t Y_local = Y / world_size_;
        
//         int64_t s0 = Y * Z;
//         int64_t s1 = Z;
//         int64_t s2 = 1;
        
//         OwnTensor::Shape transposed_shape;
//         transposed_shape.dims = {Y, X, Z};
//         OwnTensor::Tensor transposed_buf(transposed_shape, parent_tensor.tensor_.dtype(), parent_tensor.tensor_.device());
        
//         launch_fused_transpose_contiguous_kernel(
//             parent_tensor.tensor_.data<float>(),
//             transposed_buf.data<float>(),
//             X, Y, Z,
//             s0, s1, s2,
//             0, 1,
//             total_elements,
//             stream_
//         );
        
//         tensor_.reshape({{Y_local, X, Z}});
        
//         pg_->scatter_async(
//             transposed_buf.data<float>(),
//             tensor_.data<float>(),
//             shard_numel,
//             OwnTensor::Dtype::Float32,
//             root,
//             true
//         )->wait();
        
//         // Transpose back [Y_local, X, Z] -> [X, Y_local, Z]
//         int64_t rs0 = X * Z;
//         int64_t rs1 = Z;
//         int64_t rs2 = 1;
        
//         OwnTensor::Shape result_shape;
//         result_shape.dims = shape_;
//         OwnTensor::Tensor result_buf(result_shape, tensor_.dtype(), tensor_.device());
        
//         launch_fused_transpose_contiguous_kernel(
//             tensor_.data<float>(),
//             result_buf.data<float>(),
//             Y_local, X, Z,
//             rs0, rs1, rs2,
//             0, 1,
//             shard_numel,
//             stream_
//         );
        
//         tensor_ = result_buf;
//     }
    
//     setShape(tensor_.shape().dims);
// }

void launch_reverse_kernel(float* d_src, float* d_dst, int nx, int ny, int nz, int dim, cudaStream_t stream);

void DTensor::rotate3D(int dim, bool direction){
    // CRITICAL: Sync OwnTensor stream with DTensor stream to prevent race conditions
    #ifdef WITH_CUDA
    OwnTensor::cuda::setCurrentStream(stream_);
    #endif
    
    // OwnTensor::TensorOptions opts;
    // opts.dtype = OwnTensor::Dtype::Float32;
    // opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_);
    // OwnTensor::Shape shape{shape_};
    // OwnTensor::Tensor rtensor_({shape}, opts);
    // std::cout<<"\n Rotate3D "<<rank_<<"\n DTensor shape "<<rank_<<" ";
    // std::cout<<"["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] \n";
    // std::cout<<"\n Tensor shape "<<rank_<<" ";
    // std::cout<<"["<<tensor_.shape().dims[0]<<", "<<tensor_.shape().dims[1]<<", "<<tensor_.shape().dims[2]<<"] \n\n";

    int64_t nx = tensor_.shape().dims[0], ny = tensor_.shape().dims[1], nz = tensor_.shape().dims[2];

    int64_t total_elements = (int64_t)nx * ny * nz;
 
    float* d_src = static_cast<float*>(tensor_.data());
    float* d_dst;

    cudaMalloc(&d_dst, total_elements * sizeof(float));

    if (dim == 0)      {
        tensor_ = tensor_.transpose(1, 2).contiguous();
        // std::cout<<"\n after transpose "<<rank_<<"\n";
        // tensor_ = tensor_.contiguous()
        // std::cout<<"\n after contiguous "<<rank_<<"\n";
        direction ? (launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 1, stream_)):(launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 2, stream_));
        if(layout_.is_sharded()) { if (layout_.get_shard_dim() == 1 ) { layout_.set_shard_dim(2); }; if (layout_.get_shard_dim() == 2 ) {  layout_.set_shard_dim(1); }; }
    }
    else if (dim == 1) {
        tensor_ = tensor_.transpose(0, 2).contiguous();
        // std::cout<<"\n after transpose "<<rank_<<"\n";
        // tensor_ = tensor_.contiguous();
        // std::cout<<"\n after contiguous "<<rank_<<"\n";
        direction ? (launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 0, stream_)):(launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 2, stream_));
        if(layout_.is_sharded()) { if (layout_.get_shard_dim() == 0 ) { layout_.set_shard_dim(2); }; if (layout_.get_shard_dim() == 2 ) {  layout_.set_shard_dim(0); }; }
    }
    
    else {
        tensor_ = tensor_.transpose(0, 1).contiguous();
        // std::cout<<"\n after transpose "<<rank_<<"\n";
        // tensor_ = tensor_.contiguous();
        // std::cout<<"\n after contiguous "<<rank_<<"\n";
        direction ? (launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 0, stream_)):(launch_reverse_kernel(d_src, d_dst, nx, ny, nz, 1, stream_));
        if(layout_.is_sharded()) { if (layout_.get_shard_dim() == 0 ) { layout_.set_shard_dim(1); }; if (layout_.get_shard_dim() == 1 ) {  layout_.set_shard_dim(0); }; }
    }
    cudaMemcpyAsync(d_src, d_dst, total_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
    cudaStreamSynchronize(stream_);
    cudaFree(d_dst);

    // std::cout<<"\n DTensor shape "<<rank_<<"\n";
    // std::cout<<"["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] \n";
    // std::cout<<"\n Tensor shape "<<rank_<<"\n";
    // std::cout<<"["<<tensor_.shape().dims[0]<<", "<<tensor_.shape().dims[1]<<", "<<tensor_.shape().dims[2]<<"] \n";
    
    setShape(tensor_.shape().dims);
    // std::cout<<"\n DTensor shape "<<rank_<<"\n";
    // std::cout<<"["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] \n";
    // std::cout<<"\n Tensor shape "<<rank_<<"\n";
    // std::cout<<"["<<tensor_.shape().dims[0]<<", "<<tensor_.shape().dims[1]<<", "<<tensor_.shape().dims[2]<<"] \n";
    layout_.set_global_shape(shape_);

    
    // std::cout<<"\n Rotate3D post "<<rank_<<"\n DTensor shape "<<rank_<<" ";
    // std::cout<<"["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] \n";
    // std::cout<<"\n Tensor shape "<<rank_<<" ";
    // std::cout<<"["<<tensor_.shape().dims[0]<<", "<<tensor_.shape().dims[1]<<", "<<tensor_.shape().dims[2]<<"] \n\n";
    // rtensor_.reset();

    // rtensor_.release();
    
}

// Memory-optimized version of rotate3D - avoids memory leaks
void DTensor::rotate3D_mem(int dim, bool direction){
    // CRITICAL: Sync OwnTensor stream with DTensor stream to prevent race conditions
    #ifdef WITH_CUDA
    OwnTensor::cuda::setCurrentStream(stream_);
    #endif
    
    int64_t nx = shape_[0], ny = shape_[1], nz = shape_[2];

    

    int64_t total_elements = (int64_t)nx * ny * nz;
    


    // Perform transpose (creates a view, no new memory allocation)
    OwnTensor::Tensor transposed;
    if (dim == 0) {
        transposed = tensor_.transpose(1, 2);
        if(layout_.is_sharded()) { 
            if (layout_.get_shard_dim() == 1) { layout_.set_shard_dim(2); } 
            else if (layout_.get_shard_dim() == 2) { layout_.set_shard_dim(1); } 
        }
    }
    else if (dim == 1) {
        transposed = tensor_.transpose(0, 2);
        if(layout_.is_sharded()) { 
            if (layout_.get_shard_dim() == 0) { layout_.set_shard_dim(2); } 
            else if (layout_.get_shard_dim() == 2) { layout_.set_shard_dim(0); } 
        }
    }
    else {
        transposed = tensor_.transpose(0, 1);
        if(layout_.is_sharded()) { 
            if (layout_.get_shard_dim() == 0) { layout_.set_shard_dim(1); } 
            else if (layout_.get_shard_dim() == 1) { layout_.set_shard_dim(0); } 
        }
    }
    
    // Create contiguous copy - this allocates new GPU memory
    // Create contiguous copy - this allocates new GPU memory
    // Get dimensions and strides from the transposed view
    std::vector<int64_t> src_dims = transposed.shape().dims;
    std::vector<int64_t> src_strides = transposed.stride().strides;
    
    int64_t new_nx = src_dims[0];
    int64_t new_ny = src_dims[1];
    int64_t new_nz = src_dims[2];

    // Create result tensor (contiguous)
    OwnTensor::Shape result_shape;
    result_shape.dims = src_dims;
    OwnTensor::Tensor result_tensor(result_shape, tensor_.dtype(), tensor_.device());
    float* d_dst = static_cast<float*>(result_tensor.data());
    float* d_src = static_cast<float*>(transposed.data());

    // Prepare metadata for kernel
    // int64_t *d_dims, *d_strides;
    // cudaMalloc(&d_dims, 3 * sizeof(int64_t));
    // cudaMalloc(&d_strides, 3 * sizeof(int64_t));
    
    // cudaMemcpyAsync(d_dims, src_dims.data(), 3 * sizeof(int64_t), cudaMemcpyHostToDevice, stream_);
    // cudaMemcpyAsync(d_strides, src_strides.data(), 3 * sizeof(int64_t), cudaMemcpyHostToDevice, stream_);

    // Determine which axis to reverse based on rotation parameters
    int axis_to_reverse;
    if (dim == 0)      axis_to_reverse = direction ? 1 : 2;
    else if (dim == 1) axis_to_reverse = direction ? 0 : 2;
    else               axis_to_reverse = direction ? 0 : 1;
    
    // Launch the fused kernel
    launch_fused_rotate_kernel(d_src, d_dst, src_strides[0], src_strides[1], src_strides[2], 3, total_elements, new_nx, new_ny, new_nz, axis_to_reverse, stream_);
    
    // Cleanup metadata
    // cudaFree(d_dims);
    // cudaFree(d_strides);
    
    // Update tensor_ to point to the new contiguous result
    tensor_ = result_tensor;
    
    // Update shape metadata    `
    setShape(tensor_.shape().dims);
    layout_.set_global_shape(shape_);
}

// void DTensor::rotate3D(int dim, bool direction) {
//     // 1. Perform Transpose (Swaps metadata)
//     if (dim == 0)      tensor_ = tensor_.transpose(1, 2);
//     else if (dim == 1) tensor_ = tensor_.transpose(0, 2);
//     else               tensor_ = tensor_.transpose(0, 1);

//     // 2. Materialize contiguous data (Physically reorders data based on transpose)
//     // This is now our source for the reverse kernel




//     OwnTensor::Tensor rtensor_ = tensor_.contiguous();
//     float* d_src = static_cast<float*>(rtensor_.data());
    
//     // 3. Get actual physical dimensions after transpose
//     int64_t nx = rtensor_.shape().dims[0];
//     int64_t ny = rtensor_.shape().dims[1];
//     int64_t nz = rtensor_.shape().dims[2];
//     int64_t total_elements = nx * ny * nz;

//     // 4. Set up GPU pointers

//     float* d_dst;
//     cudaMalloc(&d_dst, total_elements * sizeof(float));

//     // 5. Logic for which axis to reverse to achieve 90-degree rotation
//     // After transpose(i, j), reversing one of those axes results in 90-deg rotation
//     int axis_to_reverse;
//     if (dim == 0)      axis_to_reverse = direction ? 1 : 2;
//     else if (dim == 1) axis_to_reverse = direction ? 0 : 2;
//     else               axis_to_reverse = direction ? 0 : 1;

//     launch_reverse_kernel(d_src, d_dst, nx, ny, nz, axis_to_reverse, stream_);

//     // 6. Copy back to rtensor_ memory and cleanup
//     cudaMemcpyAsync(d_src, d_dst, total_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
//     cudaStreamSynchronize(stream_);
//     cudaFree(d_dst);

//     // 7. Update DTensor metadata so it matches rtensor_
//     tensor_ = rtensor_;
//     shape_ = rtensor_.shape().dims;
//     layout_.set_global_shape(shape_);

//     rtensor_.reset();

//     rtensor_.release();
// }


void DTensor::shard_default(int dim, int root, DTensor &parent_tensor) {
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
    // std::cout<<"\n Shard Default \n DTensor shape "<<rank_<<" ";
    // std::cout<<" In Shard ["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] \n";
    // std::cout<<"\n Tensor shape "<<rank_<<" ";
    // std::cout<<" In Shard ["<<tensor_.shape().dims[0]<<", "<<tensor_.shape().dims[1]<<", "<<tensor_.shape().dims[2]<<"] \n";
    // Calculate sharded shape based on dim
    // std::vector<int64_t> local_shape = parent_tensor.shape_;
    // if (dim >= 0 && dim < local_shape.size()) {
    //     local_shape[dim] /= world_size_;
    // }
    // shape_ = local_shape;
    // std::vector<int64_t> local_shape = ;
    // tensor_.reshape({{shape_}});
    // Reallocate tensor_ to match new shape to prevent buffer overflow
    // OwnTensor::Shape new_shape;
    // new_shape.dims = shape_;
    // tensor_ = OwnTensor::Tensor(new_shape, tensor_.dtype(), tensor_.device(), tensor_.requires_grad()); 
    size_t shard_numel = 1;
    for (int d : shape_) shard_numel *= d;

    // Write scatter results directly to tensor_  
    pg_->scatter_async(
    parent_tensor.tensor_.data<float>() ,
    parent_tensor.tensor_.data<float>() + rank_ * shard_numel,
    shard_numel ,
    OwnTensor::Dtype::Float32,
    root,
    true
    )->wait();

    cudaMemcpyAsync(
        tensor_.data<float>(),
        parent_tensor.tensor_.data<float>() + rank_ * shard_numel,
        shard_numel * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream_
    );

    cudaStreamSynchronize(stream_);

    
    // std::cout<<"\n Inside Shard default \n";
    // std::cout<<"\n DTensor shape "<<rank_<<" ";
    // std::cout<<" In Shard post ["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] \n";
    // std::cout<<"\n Tensor shape "<<rank_<<" ";
    // std::cout<<" In Shard post ["<<tensor_.shape().dims[0]<<", "<<tensor_.shape().dims[1]<<", "<<tensor_.shape().dims[2]<<"] \n";
    // tensor_.display();

    // Note: Don't release parent tensor here - it will be freed when it goes out of scope
    // Calling release() here causes memory corruption when parent_tensor destructor runs
    parent_tensor.tensor_.reset();
}

void DTensor::shard_fused_transpose(int dim, int root, DTensor &parent_tensor) {
    // Shard using custom kernel to extract non-contiguous slices
    // Supports dim 0, 1, 2 for 3D tensors
    // Uses separate send_buffer to avoid race conditions
    
    #ifdef WITH_CUDA
    OwnTensor::cuda::setCurrentStream(stream_);
    #endif
    
    std::vector<int64_t> parent_shape = parent_tensor.shape_;
    
    if (parent_shape.size() != 3) {
        throw std::runtime_error("shard_fused_transpose currently only supports 3D tensors");
    }
    
    int64_t D0 = parent_shape[0];
    int64_t D1 = parent_shape[1];
    int64_t D2 = parent_shape[2];
    
    // Calculate local dimension size and shard numel
    int64_t local_dim_size;
    size_t shard_numel;
    
    if (dim == 0) {
        local_dim_size = D0 / world_size_;
        shard_numel = local_dim_size * D1 * D2;
    } else if (dim == 1) {
        local_dim_size = D1 / world_size_;
        shard_numel = D0 * local_dim_size * D2;
    } else if (dim == 2) {
        local_dim_size = D2 / world_size_;
        shard_numel = D0 * D1 * local_dim_size;
    } else {
        throw std::runtime_error("shard_fused_transpose: dim must be 0, 1, or 2");
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
            send_shape.dims = {D0 * D1 * D2};
            OwnTensor::Tensor send_buffer(send_shape, parent_tensor.tensor_.dtype(), parent_tensor.tensor_.device());
            
            nvtxRangePush("shard_fused_transpose");
            for (int r = 0; r < world_size_; r++) {
                if (dim == 1) {
                    launch_shard_dim1_kernel(
                        parent_tensor.tensor_.data<float>(),
                        send_buffer.data<float>() + r * shard_numel,
                        D0, D1, D2,
                        local_dim_size,
                        r,
                        shard_numel,
                        stream_
                    );
                } else { // dim == 2
                    launch_shard_dim2_kernel(
                        parent_tensor.tensor_.data<float>(),
                        send_buffer.data<float>() + r * shard_numel,
                        D0, D1, D2,
                        local_dim_size,
                        r,
                        shard_numel,
                        stream_
                    );
                }
            }
            cudaStreamSynchronize(stream_);
            nvtxRangePop();
            
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

// void DTensor::shard_own_transpose(int dim, int root, DTensor &parent_tensor) {
//     // Shard using OwnTensor's transpose to make target dim contiguous
//     // Only supports dim 2 for now
    
//     #ifdef WITH_CUDA
//     OwnTensor::cuda::setCurrentStream(stream_);
//     #endif
    
//     std::vector<int64_t> parent_shape = parent_tensor.shape_;
    
//     if (dim != 2 || parent_shape.size() != 3) {
//         throw std::runtime_error("shard_own_transpose currently only supports dim=2 on 3D tensors");
//     }
    
//     int64_t B = parent_shape[0];
//     int64_t C = parent_shape[1];
//     int64_t F = parent_shape[2];
//     int64_t F_local = F / world_size_;
//     size_t shard_numel = B * C * F_local;
//     int64_t total_elements = B * C * F;
    
//     // Step 1: Fused transpose [B, C, F] -> [F, C, B] to make shard dim contiguous
//     int64_t s0 = C * F, s1 = F, s2 = 1;
    
//     // OwnTensor::Shape transposed_shape;
//     // transposed_shape.dims = {F, C, B};
//     // OwnTensor::Tensor transposed_buf(transposed_shape, parent_tensor.tensor_.dtype(), parent_tensor.tensor_.device());
    
//     // launch_fused_transpose_contiguous_kernel(
//     //     parent_tensor.tensor_.data<float>(),
//     //     transposed_buf.data<float>(),
//     //     B, C, F,
//     //     s0, s1, s2,
//     //     0, 2,
//     //     total_elements,
//     //     stream_
//     // );
    
//     // Step 2: Allocate temp buffer for receiving scattered transposed data [F_local, C, B]
//     // OwnTensor::Shape recv_shape;
//     // recv_shape.dims = {F_local, C, B};
//     // OwnTensor::Tensor recv_buf(recv_shape, tensor_.dtype(), tensor_.device());
    
//     // Step 3: Scatter (now contiguous on dim 0 of transposed)
//     pg_->scatter_async(
//         transposed_buf.data<float>(),
//         recv_buf.data<float>(),
//         shard_numel,
//         OwnTensor::Dtype::Float32,
//         root,
//         true
//     )->wait();
    
//     // Step 4: Fused transpose back [F_local, C, B] -> [B, C, F_local]
//     int64_t rs0 = C * B, rs1 = B, rs2 = 1;
    
//     launch_fused_transpose_contiguous_kernel(
//         recv_buf.data<float>(),
//         tensor_.data<float>(),
//         F_local, C, B,
//         rs0, rs1, rs2,
//         0, 2,
//         shard_numel,
//         stream_
//     );
    
//     cudaStreamSynchronize(stream_);
// }



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
    pg_->all_reduce_async(tensor_.data<float>(), tensor_.data<float>(), size_, OwnTensor::Dtype::Float32, sum, false)->wait();
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
        // std:: cout<<"A"<< "["<<A.layout_.get_global_shape()[0]<<", "<<A.layout_.get_global_shape()[1]<<", "<<A.layout_.get_global_shape()[2]<<"] "<<std::endl;
        // std:: cout<<"B"<< "["<<B.layout_.get_global_shape()[0]<<", "<<B.layout_.get_global_shape()[1]<<", "<<B.layout_.get_global_shape()[2]<<"] "<<std::endl;
    
        // std:: cout<<"shape"<< "["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] "<<std::endl;

        throw std::runtime_error("DTensor shape doesnt match matmul output shape ");
    }
    std:: cout<< "\n Matmul \n";
    // std:: cout<< "["<<A.layout_.get_global_shape()[0]<<", "<<A.layout_.get_global_shape()[1]<<", "<<A.layout_.get_global_shape()[2]<<"] "<<std::endl;
    // std:: cout<< "["<<B.layout_.get_global_shape()[0]<<", "<<B.layout_.get_global_shape()[1]<<", "<<B.layout_.get_global_shape()[2]<<"] "<<std::endl;
    
    // std:: cout<< "["<<shape_[0]<<", "<<shape_[1]<<", "<<shape_[2]<<"] "<<std::endl;
    
    tensor_ = TensorOpsBridge::matmul(A.tensor_, B.tensor_);
}

void DTensor::Linear(  DTensor& Input,  DTensor& Weights,  DTensor& Bias) {

    tensor_ = mlp_forward::linear(Input.tensor_, Weights.tensor_, Bias.tensor_);

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

ag::Value& DTensor::get_value(){ return value_; }

// void DTensor::enable_grad() {
//     // If value_ is stale (pointing to different data than tensor_), recreate it
//     if (!value_.node || value_.val().data() != tensor_.data()) {
//         // std::cout << "DTensor::enable_grad: Refreshing value_ node data sync.\n";
//         value_ = ag::make_tensor(tensor_, "");
//     }
    
//     if (value_.node) {
//         value_.node->requires_grad_flag_ = true;
        
//         if (value_.node->grad.data() == nullptr) {
//              OwnTensor::TensorOptions opts;
//              opts.dtype = tensor_.dtype();
//              opts.device = tensor_.device();
//              value_.node->grad = OwnTensor::Tensor::zeros(tensor_.shape(), opts);
//         }
//     }
// }

void DTensor::display(){
    tensor_.to_cpu().display();
}

void DTensor::rand() {
    OwnTensor::TensorOptions opts;
        opts.dtype = OwnTensor::Dtype::Float32;
        opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_);
    tensor_  = OwnTensor::Tensor::rand({shape_}, opts, 0.0f, 0.1f);
}