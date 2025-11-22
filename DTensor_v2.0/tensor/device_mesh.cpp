#include "tensor/device_mesh.h"
#include <algorithm>
#include <cassert>

// =========================================================
// Constructor
// =========================================================
DeviceMesh::DeviceMesh(const std::vector<int>& mesh_shape, 
                       const std::vector<int>& device_ids)
    : mesh_shape_(mesh_shape) {
    
    if (mesh_shape_.empty()) {
        throw std::runtime_error("DeviceMesh: mesh_shape cannot be empty");
    }
    
    // Calculate total number of devices
    total_devices_ = std::accumulate(mesh_shape_.begin(), mesh_shape_.end(), 
                                     1, std::multiplies<int>());
    
    // Get global rank from MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank_);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (total_devices_ != world_size) {
        throw std::runtime_error("DeviceMesh: mesh size (" + 
                                std::to_string(total_devices_) + 
                                ") must match MPI world size (" + 
                                std::to_string(world_size) + ")");
    }
    
    // Setup device IDs
    if (device_ids.empty()) {
        // Default: device_ids = [0, 1, 2, ..., N-1]
        device_ids_.resize(total_devices_);
        std::iota(device_ids_.begin(), device_ids_.end(), 0);
    } else {
        if ((int)device_ids.size() != total_devices_) {
            throw std::runtime_error("DeviceMesh: device_ids size must match total devices");
        }
        device_ids_ = device_ids;
    }
    
    // Compute this rank's coordinate in the mesh
    my_coordinate_ = get_coordinate(global_rank_);
    
    // Initialize process groups for each mesh dimension
    initialize_process_groups();
}

DeviceMesh::~DeviceMesh() {
    // ProcessGroup destructors will handle cleanup
}

// =========================================================
// Coordinate Mapping
// =========================================================
std::vector<int> DeviceMesh::get_coordinate(int rank) const {
    std::vector<int> coord(ndim());
    int remaining = rank;
    
    // Row-major ordering: rightmost dimension varies fastest
    for (int dim = ndim() - 1; dim >= 0; --dim) {
        coord[dim] = remaining % mesh_shape_[dim];
        remaining /= mesh_shape_[dim];
    }
    
    return coord;
}

int DeviceMesh::get_rank(const std::vector<int>& coordinate) const {
    if ((int)coordinate.size() != ndim()) {
        throw std::runtime_error("DeviceMesh: coordinate size must match mesh ndim");
    }
    
    int rank = 0;
    int stride = 1;
    
    // Row-major ordering
    for (int dim = ndim() - 1; dim >= 0; --dim) {
        if (coordinate[dim] < 0 || coordinate[dim] >= mesh_shape_[dim]) {
            throw std::runtime_error("DeviceMesh: coordinate out of bounds");
        }
        rank += coordinate[dim] * stride;
        stride *= mesh_shape_[dim];
    }
    
    return rank;
}

// =========================================================
// Process Group Management
// =========================================================
std::vector<int> DeviceMesh::get_group_ranks(int mesh_dim) const {
    if (mesh_dim < 0 || mesh_dim >= ndim()) {
        throw std::runtime_error("DeviceMesh: invalid mesh_dim");
    }
    
    std::vector<int> ranks;
    
    // All ranks that differ only in mesh_dim coordinate
    // Example: 2D mesh [2, 2], rank 0 at [0, 0], mesh_dim=0
    // -> group ranks are [0, 2] (coords [0,0] and [1,0])
    
    std::vector<int> base_coord = my_coordinate_;
    
    for (int i = 0; i < mesh_shape_[mesh_dim]; ++i) {
        base_coord[mesh_dim] = i;
        ranks.push_back(get_rank(base_coord));
    }
    
    return ranks;
}

int DeviceMesh::get_dim_rank(int mesh_dim) const {
    if (mesh_dim < 0 || mesh_dim >= ndim()) {
        throw std::runtime_error("DeviceMesh: invalid mesh_dim");
    }
    return my_coordinate_[mesh_dim];
}

void DeviceMesh::initialize_process_groups() {
    process_groups_.resize(ndim());
    
    for (int mesh_dim = 0; mesh_dim < ndim(); ++mesh_dim) {
        // Get ranks in this dimension's process group
        std::vector<int> group_ranks = get_group_ranks(mesh_dim);
        int group_size = group_ranks.size();
        
        // Find my position in this group
        auto it = std::find(group_ranks.begin(), group_ranks.end(), global_rank_);
        int my_group_rank = std::distance(group_ranks.begin(), it);
        
        // Create NCCL unique ID (use first rank in group as root)
        ncclUniqueId nccl_id = create_nccl_id(group_ranks[0]);
        
        // Get CUDA device for this rank
        int device = device_ids_[global_rank_];
        
        // Create ProcessGroup for this mesh dimension
        // Note: ProcessGroup expects (rank_in_group, group_size, device, nccl_id)
        process_groups_[mesh_dim] = std::make_shared<ProcessGroup>(
            my_group_rank, group_size, device, nccl_id
        );
    }
}

ncclUniqueId DeviceMesh::create_nccl_id(int root_rank) {
    ncclUniqueId nccl_id;
    
    if (global_rank_ == root_rank) {
        ncclGetUniqueId(&nccl_id);
    }
    
    // Broadcast the NCCL ID to all ranks
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, root_rank, MPI_COMM_WORLD);
    
    return nccl_id;
}

std::shared_ptr<ProcessGroup> DeviceMesh::get_process_group(int mesh_dim) {
    if (mesh_dim < 0 || mesh_dim >= ndim()) {
        throw std::runtime_error("DeviceMesh: invalid mesh_dim");
    }
    return process_groups_[mesh_dim];
}

// =========================================================
// Accessors
// =========================================================
int DeviceMesh::size() const {
    return total_devices_;
}

// =========================================================
// Debug
// =========================================================
void DeviceMesh::describe() const {
    std::ostringstream oss;
    oss << "[DeviceMesh] Rank " << global_rank_ << "/" << total_devices_ << " | Shape: [";
    for (size_t i = 0; i < mesh_shape_.size(); ++i) {
        oss << mesh_shape_[i];
        if (i < mesh_shape_.size() - 1) oss << ", ";
    }
    oss << "] | Coordinate: [";
    for (size_t i = 0; i < my_coordinate_.size(); ++i) {
        oss << my_coordinate_[i];
        if (i < my_coordinate_.size() - 1) oss << ", ";
    }
    oss << "]";
    
    std::cout << oss.str() << std::endl;
}
