// #pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "process_group/ProcessGroupNCCL_new.h"
#include "memory/cachingAllocator.hpp"


#include "bridge/tensor_ops_bridge.h"
#include "device/Device.h"
#include "dtype/Dtype.h"

#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include "tensor/placement.h"
#include "reverse.cuh"

#include "ad/ag_all.hpp"  

using namespace OwnTensor;


extern CachingAllocator gAllocator;

class DTensor {
public:

    DTensor(const DeviceMesh& device_mesh, std::shared_ptr<ProcessGroupNCCL> pg, Layout layout);
    ~DTensor();

    void setData(const std::vector<float>& host_data) ;
    std::vector<float> getData() const; 
    
    // DTensor add(const DTensor& other) const;
    // DTensor sub(const DTensor& other) const;
    // DTensor mul(const DTensor& other) const;
    // DTensor div(const DTensor& other) const;
    
    void matmul( DTensor& A,  DTensor& B);

    void Linear(  DTensor& Input,  DTensor& Weights,  DTensor& Bias);
    // DTensor matmul(const DTensor& other) const;
    // DTensor  _column_parallel_matmul(const DTensor& other) const;
    // DTensor _row_parallel_matmul(const DTensor& other) const;
    // DTensor reshape(const std::vector<int64_t>& new_global_shape) const;


    void replicate(int root = 0);



    void rotate3D( int dim, bool direction);
    void rotate3D_mem( int dim, bool direction);  // Memory-optimized version
    
    void shard(int dim, int root , DTensor &parent_tensor );

    void shard_transpose(int dim, int root, DTensor &parent_tensor);
    // void shard_transpose_fused(int dim, int root, DTensor &parent_tensor);  // Fused transpose+contiguous

    void shard_default(int dim, int root, DTensor &parent_tensor);
    void shard_fused_transpose(int dim, int root, DTensor &parent_tensor);  // Uses custom kernel for reordering
    // void shard_own_transpose(int dim, int root, DTensor &parent_tensor);    // Uses OwnTensor transpose

    void sync();

    void assemble(int dim, int root, DTensor &sharded_tensor ); 

    void permute_striped(int dim = 0);



    // void DTensor::qkvsplit( DTensor &q, DTensor &k,DTensor &v);
    void unpermute_striped(int dim = 0);

    // void saveCheckpoint(const std::string& path) const;
    // void loadCheckpoint(const std::string& path);
    void display();

    void rand() ;

    void print() const;

    ag::Value& get_value();
    // void enable_grad();
    // void get_tensor(){ return tensor_; }
    
    void setShape(std::vector<int64_t>newShape){ shape_ = newShape ; }
    const Layout& get_layout()  { return layout_; }
    const OwnTensor::Tensor& local_tensor() const { return tensor_; }
    std::shared_ptr<ProcessGroupNCCL> get_pg() const { return pg_; }
    const DeviceMesh& get_device_mesh() const { return device_mesh_; }
    int rank() const { return rank_; }
    int getSize() const { return size_;}

private:

    // DTensor(DeviceMesh device_mesh,
    //         std::shared_ptr<ProcessGroup> pg,
    //         const Layout& layout);


    // DTensor _column_parallel_matmul(const DTensor& other) const;
    // DTensor _row_parallel_matmul(const DTensor& other) const;


    int rank_;
    int world_size_;
    const DeviceMesh& device_mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    cudaStream_t stream_;
    ag::Value value_ ;

    Layout layout_;
    
    OwnTensor::Tensor tensor_;      
    // OwnTensor::Tensor temp_tensor_; 


    int size_;
    std::vector<int64_t> shape_; 
    std::string dtype_ = "float32";
    Block* data_block_;
    Block* temp_block_;



    void printRecursive(const std::vector<float>& data,
                        const std::vector<int64_t>& dims,
                        int dim,
                        int offset) const;
};

// void launch_reverse_kernel(float* d_src, float* d_dst, int nx, int ny, int nz, int dim, cudaStream_t stream);