// #pragma once
// #include <vector>
// #include <memory>
// #include <iostream>
// #include <string>
// #include "process_group.h"

// class DTensor {
// public:
//     DTensor(int rank, int world_size, ProcessGroup* pg);
//     ~DTensor();

//     void allReduce();
//     void reduceScatter();
//     void allGather();
//     void broadcast(int root);

//     void print() const;
//     int size() const { return size_; }

//     void setData(const std::vector<float>& data);
//     std::vector<float> getData() const;

//     // ------------------- NEW METHODS -------------------
//     void saveCheckpoint(const std::string& path) const;
//     void loadCheckpoint(const std::string& path);
//     // ---------------------------------------------------

// private:
//     int rank_;
//     int world_size_;
//     int size_;
//     float* data_;
//     float* temp_buf_;
//     ProcessGroup* pg_;

//     // ------------------- NEW METADATA -------------------
//     int shape_[1];             // Currently 1D tensor
//     std::string dtype_ = "float32";
//     // ---------------------------------------------------
// };


#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "process_group.h"
#include "cachingAllocator.hpp"

// Global allocator declaration
extern CachingAllocator gAllocator;

class DTensor {
public:
    DTensor(int rank, int world_size, ProcessGroup* pg);
    ~DTensor();

    void allReduce();
    void reduceScatter();
    void allGather();
    void broadcast(int root);
    void print() const;

    void setData(const std::vector<float>& data);
    std::vector<float> getData() const;

    // Checkpointing
    void saveCheckpoint(const std::string& path) const;
    void loadCheckpoint(const std::string& path);

private:
    int rank_;
    int world_size_;
    int size_;
    float* data_;
    float* temp_buf_;
    ProcessGroup* pg_;

    cudaStream_t stream_; // Each DTensor uses its own stream

    Block* data_block_;
    Block* temp_block_;

    int shape_[1];
    std::string dtype_ = "float32";
};
