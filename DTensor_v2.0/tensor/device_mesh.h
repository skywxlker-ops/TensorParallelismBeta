#pragma once
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include "process_group/process_group.h"
#include <mpi.h>
#include <nccl.h>



class DeviceMesh {
public:

    DeviceMesh(const std::vector<int>& mesh_shape, 
               const std::vector<int>& device_ids = {});
    
    ~DeviceMesh();
    
    std::vector<int> get_coordinate(int rank) const;
    
    
    int get_rank(const std::vector<int>& coordinate) const;

 
    std::shared_ptr<ProcessGroup> get_process_group(int mesh_dim);
 
    std::vector<int> get_group_ranks(int mesh_dim) const;
    
    
    int get_dim_rank(int mesh_dim) const;
    
    
    int ndim() const { return mesh_shape_.size(); }
    const std::vector<int>& shape() const { return mesh_shape_; }
    int size() const;  // Total number of devices
    int rank() const { return global_rank_; }
    int world_size() const { return total_devices_; }

    void describe() const;
    
private:
    std::vector<int> mesh_shape_;      
    std::vector<int> device_ids_;      
    int total_devices_;              
    int global_rank_;                  
    std::vector<int> my_coordinate_;   
    
    std::vector<std::shared_ptr<ProcessGroup>> process_groups_;
    
    
    std::vector<MPI_Comm> mpi_comms_;

    void initialize_process_groups();
    
    ncclUniqueId create_nccl_id(int root_rank, MPI_Comm comm);
};
