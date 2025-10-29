#pragma once
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include "../process_group/process_group.h"

class DTensor {
public:
    DTensor(int world_size, int slice_size, int rank, ProcessGroup* pg);
    ~DTensor();

    float* deviceData();
    std::vector<float>& hostData();
    size_t size() const;

    void copyDeviceToHost();
    void allReduce();
    void reduceScatter();
    void allGather();
    void broadcast();

    std::string toString();

private:
    int world_size_, slice_size_, rank_;
    std::vector<float> h_data_;
    float* d_data_ = nullptr;
    ProcessGroup* pg_;
};
