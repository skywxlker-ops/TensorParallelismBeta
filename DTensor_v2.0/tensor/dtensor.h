#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "process_group.h"
#include "memory/cachingAllocator.hpp"


#include "bridge/tensor_ops_bridge.h"
#include "device/Device.h"
#include "dtype/Dtype.h"


#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include "tensor/placement.h"


using namespace OwnTensor;


extern CachingAllocator gAllocator;

class DTensor {
public:

    DTensor(std::shared_ptr<DeviceMesh> device_mesh, std::shared_ptr<ProcessGroup> pg);
    ~DTensor();


    void allReduce();
    void reduceScatter();
    void allGather();
    void broadcast(int root);


    void setData(const std::vector<float>& host_data, const Layout& layout);
    std::vector<float> getData() const; 

    DTensor add(const DTensor& other) const;
    DTensor sub(const DTensor& other) const;
    DTensor mul(const DTensor& other) const;
    DTensor div(const DTensor& other) const;
    DTensor matmul(const DTensor& other) const;

    DTensor reshape(const std::vector<int>& new_global_shape) const;

    void saveCheckpoint(const std::string& path) const;
    void loadCheckpoint(const std::string& path);


    void print() const;
    

    const Layout& get_layout() const { return layout_; }
    const OwnTensor::Tensor& local_tensor() const { return tensor_; }
    std::shared_ptr<ProcessGroup> get_pg() const { return pg_; }
    std::shared_ptr<DeviceMesh> get_device_mesh() const { return device_mesh_; }
    int rank() const { return rank_; }


private:

    DTensor(std::shared_ptr<DeviceMesh> device_mesh,
            std::shared_ptr<ProcessGroup> pg,
            const OwnTensor::Tensor& local_tensor,
            const Layout& layout);


    DTensor _column_parallel_matmul(const DTensor& other) const;
    DTensor _row_parallel_matmul(const DTensor& other) const;


    int rank_;
    int world_size_;
    std::shared_ptr<DeviceMesh> device_mesh_;
    std::shared_ptr<ProcessGroup> pg_;
    cudaStream_t stream_;

    Layout layout_;
    
    OwnTensor::Tensor tensor_;      
    OwnTensor::Tensor temp_tensor_; 


    int size_;
    std::vector<int> shape_; 
    std::string dtype_ = "float32";
    Block* data_block_;
    Block* temp_block_;

    
    void printRecursive(const std::vector<float>& data,
                        const std::vector<int>& dims,
                        int dim,
                        int offset) const;
};