// // #pragma once
// // #include <vector>
// // #include <memory>
// // #include <iostream>
// // #include <string>
// // #include <cuda_runtime.h>
// // #include "process_group.h"
// // #include "cachingAllocator.hpp"

// // // --- Tensor & Ops Integration ---
// // #include "core/Tensor.h"
// // #include "ops/TensorOps.h"
// // #include "device/Device.h"
// // #include "dtype/Dtype.h"

// // using namespace OwnTensor;

// // // Global allocator declaration
// // extern CachingAllocator gAllocator;

// // class DTensor {
// // public:
// //     DTensor(int rank, int world_size, ProcessGroup* pg);
// //     ~DTensor();

// //     // Distributed operations
// //     void allReduce();
// //     void reduceScatter();
// //     void allGather();
// //     void broadcast(int root);

// //     // Data interface
// //     void setData(const std::vector<float>& data);
// //     std::vector<float> getData() const;

// //     // Checkpointing
// //     void saveCheckpoint(const std::string& path) const;
// //     void loadCheckpoint(const std::string& path);

// //     // Debug utility
// //     void print() const;

// // private:
// //     int rank_;
// //     int world_size_;
// //     int size_;
// //     ProcessGroup* pg_;

// //     cudaStream_t stream_; // NCCL stream

// //     Block* data_block_;
// //     Block* temp_block_;

// //     int shape_[1];
// //     std::string dtype_ = "float32";

// //     // --- TensorLib integration ---
// //     OwnTensor::Tensor tensor_;       // main tensor
// //     OwnTensor::Tensor temp_tensor_;  // temporary tensor (for collective ops)
// // };


// // dtensor.h

// #pragma once
// #include <vector>
// #include <memory>
// #include <iostream>
// #include <string>
// #include <cuda_runtime.h>
// #include "process_group.h"
// #include "memory/cachingAllocator.hpp"

// // --- Tensor & Ops Integration ---
// #include "bridge/tensor_ops_bridge.h"
// #include "device/Device.h"
// #include "dtype/Dtype.h"

// using namespace OwnTensor;

// // Global allocator declaration
// extern CachingAllocator gAllocator;

// class DTensor {
// public:
//     DTensor(int rank, int world_size, ProcessGroup* pg);
//     ~DTensor();

//     // === Distributed collectives ===
//     void allReduce();
//     void reduceScatter();
//     void allGather();
//     void broadcast(int root);

//     // === Data interface ===
//     void setData(const std::vector<float>& data);
//     std::vector<float> getData() const;

//     // === Tensor Ops (via bridge) ===
//     DTensor add(const DTensor& other) const;
//     DTensor sub(const DTensor& other) const;
//     DTensor mul(const DTensor& other) const;
//     DTensor div(const DTensor& other) const;
//     DTensor matmul(const DTensor& other) const;

//     // === View / Reshape ===
//     DTensor reshape(int rows, int cols) const;


//     // === Checkpointing ===
//     void saveCheckpoint(const std::string& path) const;
//     void loadCheckpoint(const std::string& path);

//     // === Debug utilities ===
//     void print() const;

// private:
//     int rank_;
//     int world_size_;
//     int size_;
//     ProcessGroup* pg_;
//     cudaStream_t stream_;

//     int shape_[1];
//     std::string dtype_ = "float32";

//     Block* data_block_;
//     Block* temp_block_;

//     // --- TensorLib integration ---
//     OwnTensor::Tensor tensor_;
//     OwnTensor::Tensor temp_tensor_;
// };


#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "process_group.h"
#include "memory/cachingAllocator.hpp"

// --- Tensor & Ops Integration ---
#include "bridge/tensor_ops_bridge.h"
#include "device/Device.h"
#include "dtype/Dtype.h"

using namespace OwnTensor;

// Global allocator declaration
extern CachingAllocator gAllocator;

class DTensor {
public:
    DTensor(int rank, int world_size, ProcessGroup* pg);
    ~DTensor();

    // === Distributed collectives ===
    void allReduce();
    void reduceScatter();
    void allGather();
    void broadcast(int root);

    // === Data interface ===
    void setData(const std::vector<float>& data, const std::vector<int>& shape = {});
    std::vector<float> getData() const;

    // === Tensor Ops (via bridge) ===
    DTensor add(const DTensor& other) const;
    DTensor sub(const DTensor& other) const;
    DTensor mul(const DTensor& other) const;
    DTensor div(const DTensor& other) const;
    DTensor matmul(const DTensor& other) const;

    // === View / Reshape ===
    DTensor reshape(int rows, int cols) const;
    DTensor reshape(const std::vector<int>& new_shape) const;

    // === Checkpointing ===
    void saveCheckpoint(const std::string& path) const;
    void loadCheckpoint(const std::string& path);

    // === Debug utilities ===
    void print() const;

private:
    int rank_;
    int world_size_;
    int size_;
    ProcessGroup* pg_;
    cudaStream_t stream_;

    std::vector<int> shape_;
    std::string dtype_ = "float32";

    Block* data_block_;
    Block* temp_block_;

    // --- TensorLib integration ---
    OwnTensor::Tensor tensor_;
    OwnTensor::Tensor temp_tensor_;

    // === Internal helpers ===
    void printRecursive(const std::vector<float>& data,
                        const std::vector<int>& dims,
                        int dim,
                        int offset) const;
};
