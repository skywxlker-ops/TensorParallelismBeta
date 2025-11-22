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

// =========================================================
// DeviceMesh - N-dimensional device topology
// =========================================================
// DeviceMesh represents an N-dimensional array of devices, enabling
// multi-dimensional parallelism strategies (e.g., 2D TP+DP, 3D TP+DP+PP)
class DeviceMesh {
public:
    // Constructor
    // mesh_shape: shape of the mesh, e.g., [2, 2] for 2x2 grid
    // device_ids: flattened list of device IDs (optional, defaults to [0, 1, ..., N-1])
    DeviceMesh(const std::vector<int>& mesh_shape, 
               const std::vector<int>& device_ids = {});
    
    ~DeviceMesh();
    
    // --- Coordinate Mapping ---
    
    // Get the mesh coordinates for a given global rank
    // E.g., in a [2, 2] mesh, rank 3 -> [1, 1]
    std::vector<int> get_coordinate(int rank) const;
    
    // Get global rank from mesh coordinates
    // E.g., in a [2, 2] mesh, [1, 1] -> rank 3
    int get_rank(const std::vector<int>& coordinate) const;
    
    // --- Process Groups ---
    
    // Get the process group for a specific mesh dimension
    // This process group contains all ranks that share the same coordinates
    // in all other dimensions
    std::shared_ptr<ProcessGroup> get_process_group(int mesh_dim);
    
    // Get the list of ranks in the same process group as this rank for mesh_dim
    std::vector<int> get_group_ranks(int mesh_dim) const;
    
    // Get the rank within a specific mesh dimension's process group
    int get_dim_rank(int mesh_dim) const;
    
    // --- Accessors ---
    
    int ndim() const { return mesh_shape_.size(); }
    const std::vector<int>& shape() const { return mesh_shape_; }
    int size() const;  // Total number of devices
    int rank() const { return global_rank_; }
    int world_size() const { return total_devices_; }
    
    // --- Debug ---
    void describe() const;
    
private:
    std::vector<int> mesh_shape_;      // E.g., [2, 2] for 2x2 mesh
    std::vector<int> device_ids_;      // Flattened device IDs
    int total_devices_;                // Product of mesh_shape_
    int global_rank_;                  // This process's global rank
    std::vector<int> my_coordinate_;   // This rank's position in the mesh
    
    // Process groups: one per mesh dimension
    // process_groups_[i] handles communication along mesh dimension i
    std::vector<std::shared_ptr<ProcessGroup>> process_groups_;
    
    // MPI sub-communicators: one per mesh dimension
    // mpi_comms_[i] contains ranks that share coordinates in all dims except i
    std::vector<MPI_Comm> mpi_comms_;
    
    // Helper to initialize process groups for each mesh dimension
    void initialize_process_groups();
    
    // Helper to create NCCL unique ID and broadcast it within a sub-communicator
    ncclUniqueId create_nccl_id(int root_rank, MPI_Comm comm);
};
