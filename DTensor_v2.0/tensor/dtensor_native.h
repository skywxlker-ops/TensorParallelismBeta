#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "process_group/ProcessGroupNCCL.h"
#include "memory/cachingAllocator.hpp"

// Native OwnTensor includes  
#include "core/Tensor.h"
#include "device/Device.h"
#include "dtype/Dtype.h"

#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include "tensor/placement.h"

// Forward declare
extern CachingAllocator gAllocator;

namespace OwnTensor {

// Native DTensor integrated directly into OwnTensor namespace
// Eliminates Bridge indirection for benchmarking
class DTensorNative {
public:
    DTensorNative(std::shared_ptr<DeviceMesh> device_mesh, std::shared_ptr<ProcessGroupNCCL> pg);
    ~DTensorNative();

    void setData(const std::vector<float>& host_data, const Layout& layout);
    std::vector<float> getData() const;

    // Only implement matmul for benchmark
    DTensorNative matmul(const DTensorNative& other) const;

    // Getters
    const Layout& get_layout() const { return layout_; }
    const OwnTensor::Tensor& local_tensor() const { return tensor_; }
    std::shared_ptr<ProcessGroupNCCL> get_pg() const { return pg_; }
    std::shared_ptr<DeviceMesh> get_device_mesh() const { return device_mesh_; }
    int rank() const { return rank_; }

private:
    DTensorNative(std::shared_ptr<DeviceMesh> device_mesh,
                  std::shared_ptr<ProcessGroupNCCL> pg,
                  const OwnTensor::Tensor& local_tensor,
                  const Layout& layout);

    DTensorNative _column_parallel_matmul(const DTensorNative& other) const;
    DTensorNative _row_parallel_matmul(const DTensorNative& other) const;

    int rank_;
    int world_size_;
    std::shared_ptr<DeviceMesh> device_mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    cudaStream_t stream_;

    Layout layout_;
    OwnTensor::Tensor tensor_;
    OwnTensor::Tensor temp_tensor_;

    int size_;
    std::vector<int64_t> shape_;
    std::string dtype_ = "float32";
    Block* data_block_;
    Block* temp_block_;
};

} // namespace OwnTensor
