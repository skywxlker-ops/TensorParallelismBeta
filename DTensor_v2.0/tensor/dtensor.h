#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include "process_group.h"

class DTensor {
public:
    DTensor(int rank, int world_size, ProcessGroup* pg);
    ~DTensor();

    void allReduce();
    void reduceScatter();
    void allGather();
    void broadcast(int root);

    void print() const;
    int size() const { return size_; }

    void setData(const std::vector<float>& data);
    std::vector<float> getData() const;

    // ------------------- NEW METHODS -------------------
    void saveCheckpoint(const std::string& path) const;
    void loadCheckpoint(const std::string& path);
    // ---------------------------------------------------

private:
    int rank_;
    int world_size_;
    int size_;
    float* data_;
    float* temp_buf_;
    ProcessGroup* pg_;

    // ------------------- NEW METADATA -------------------
    int shape_[1];             // Currently 1D tensor
    std::string dtype_ = "float32";
    // ---------------------------------------------------
};
