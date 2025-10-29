#pragma once
#include <vector>
#include <memory>
#include <iostream>
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


private:
    int rank_;
    int world_size_;
    int size_;
    float* data_;
    float* temp_buf_;  // used for scatter/gather
    ProcessGroup* pg_;
};
