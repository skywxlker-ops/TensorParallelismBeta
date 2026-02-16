#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "/home/blu-bridge25/Study/Code/avengersassemble/TensorParallelismBeta/DTensor/process_group/ProcessGroupNCCL.h"
// #include "memory/cachingAllocator.hpp"


// #include "bridge/tensor_ops_bridge.h"
#include "device/Device.h"
#include "dtype/Dtype.h"

#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include "tensor/placement.h"
#include "autograd/AutogradOps.h"
#include "autograd/Engine.h"

using namespace OwnTensor;


// extern CachingAllocator gAllocator;

class DTensor {
public:

    DTensor();  // Default constructor for member initialization
    DTensor(const DeviceMesh& device_mesh, std::shared_ptr<ProcessGroupNCCL> pg, Layout layout, std::string name = "", float sd = 0.02f, int seed = 42);
    ~DTensor();

    void setData(const std::vector<float>& host_data) ;
    void setData(const std::vector<int64_t>& host_data) ;
    std::vector<float> getData() const; 
    
    // DTensor add(const DTensor& other) const;
    // DTensor sub(const DTensor& other) const;
    // DTensor mul(const DTensor& other) const;
    // DTensor div(const DTensor& other) const;
    
    void matmul( DTensor& A,  DTensor& B);

    void Linear(  DTensor& Input,  DTensor& Weights,  DTensor& Bias);
    
    // Autograd-enabled linear layer for gradient tracking
    void linear_w_autograd(DTensor& Input, DTensor& Weights, DTensor& Bias);
    void linear_w_autograd(DTensor& Input, DTensor& Weights);  // No-bias overload
    
    // Backward pass - computes gradients for tensors with requires_grad=true
    void backward();


    void replicate(int root = 0);


    void shard(int dim, int root , DTensor &parent_tensor );

    void shard_fused_transpose(int dim, int root, DTensor &parent_tensor);  // Uses custom kernel for reordering
    // void shard_own_transpose(int dim, int root, DTensor &parent_tensor);    // Uses OwnTensor transpose

    void sync();              // All-reduce with wait (blocking)
    void sync_async();         // All-reduce without wait (async)
    // void sync_async_backward_hook();
    void sync_w_autograd(op_t op = sum);     // Autograd-aware sync (registers backward for gradient all-reduce)
    void wait();               // Wait for pending async collective
    // void wait_backward_hook();
    bool has_pending_collective() const;  // Check if async collective pending

    void assemble(int dim, int root, DTensor &sharded_tensor ); 

    void permute_striped(int dim = 0);

    // void DTensor::qkvsplit( DTensor &q, DTensor &k,DTensor &v);
    void unpermute_striped(int dim = 0);

    void display();

    // void rand() ;

    void print() const;

    
    void setShape(std::vector<int64_t>newShape){ shape_ = newShape ; }
    const Layout& get_layout()  { return layout_; }
    const OwnTensor::Tensor& local_tensor() const { return tensor_; }
    OwnTensor::Tensor& mutable_tensor() { return tensor_; }
    void set_tensor(OwnTensor::Tensor& tensor){ tensor_ = tensor; }
    std::shared_ptr<ProcessGroupNCCL> get_pg() const { return pg_; }
    const DeviceMesh& get_device_mesh() const { return *device_mesh_; }
    int rank() const { return rank_; }
    int getSize() const { return size_;}
    std::string name(){return name_;}

private:

    int rank_;
    int world_size_;
    const DeviceMesh* device_mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    cudaStream_t stream_;
    // ag::Value value_ ;

    Layout layout_;
    
    OwnTensor::Tensor tensor_;      
    
    int size_;
    std::vector<int64_t> shape_; 
    std::string dtype_ = "float32";
    // Block* data_block_;
    // Block* temp_block_;
    std::string name_;
    
    // Async collective tracking
    bool has_pending_collective_ = false;
    std::shared_ptr<Work> pending_work_ = nullptr;

    void printRecursive(const std::vector<float>& data,
                        const std::vector<int64_t>& dims,
                        int dim,
                        int offset) const;
};
