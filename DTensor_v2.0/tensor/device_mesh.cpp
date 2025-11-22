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
    // Free MPI sub-communicators
    for (auto& comm : mpi_comms_) {
        if (comm != MPI_COMM_NULL) {
            MPI_Comm_free(&comm);
        }
    }
    // ProcessGroup destructors will handle NCCL cleanup
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
    mpi_comms_.resize(ndim(), MPI_COMM_NULL);
    
    for (int mesh_dim = 0; mesh_dim < ndim(); ++mesh_dim) {
        // Get ranks in this dimension's process group
        std::vector<int> group_ranks = get_group_ranks(mesh_dim);
        int group_size = group_ranks.size();
        
        // Find my position in this group
        auto it = std::find(group_ranks.begin(), group_ranks.end(), global_rank_);
        int my_group_rank = std::distance(group_ranks.begin(), it);
        
        // Create MPI sub-communicator for this mesh dimension
        // Color calculation: Convert mesh coordinates (excluding mesh_dim) to a unique integer
        // Example: [2, 4] mesh, mesh_dim=1, rank 5 has coord [1, 1]
        //   -> color based on coord[0]=1 (all ranks in same column share this)
        
        int color = 0;
        int multiplier = 1;
        for (int d = ndim() - 1; d >= 0; --d) {
            if (d != mesh_dim) {
                color += my_coordinate_[d] * multiplier;
                multiplier *= mesh_shape_[d];
            }
        }
        
        // Key for ordering within the sub-communicator (use our position along mesh_dim)
        int key = my_coordinate_[mesh_dim];
        

        
        // Split MPI_COMM_WORLD to create sub-communicator
        int mpi_err = MPI_Comm_split(MPI_COMM_WORLD, color, key, &mpi_comms_[mesh_dim]);
        if (mpi_err != MPI_SUCCESS) {
            throw std::runtime_error("DeviceMesh: MPI_Comm_split failed for mesh_dim " + 
                                   std::to_string(mesh_dim));
        }
        
        // Verify the sub-communicator has the correct size
        int sub_comm_size;
        MPI_Comm_size(mpi_comms_[mesh_dim], &sub_comm_size);
        if (sub_comm_size != group_size) {
            std::cerr << "[Rank " << global_rank_ << "] ERROR: mesh_dim=" << mesh_dim
                      << " sub_comm_size=" << sub_comm_size 
                      << " expected=" << group_size << "\n";
            throw std::runtime_error("DeviceMesh: MPI sub-communicator size mismatch");
        }
        
        // Verify our rank in the sub-communicator
        int sub_comm_rank;
        MPI_Comm_rank(mpi_comms_[mesh_dim], &sub_comm_rank);
        if (sub_comm_rank != my_group_rank) {
            std::cerr << "[Rank " << global_rank_ << "] ERROR: mesh_dim=" << mesh_dim
                      << " sub_comm_rank=" << sub_comm_rank 
                      << " expected=" << my_group_rank << "\n";
            throw std::runtime_error("DeviceMesh: MPI sub-communicator rank mismatch");
        }
        
        // Create NCCL unique ID using sub-communicator
        // Root is rank 0 in the sub-communicator (not global rank)
        ncclUniqueId nccl_id = create_nccl_id(0, mpi_comms_[mesh_dim]);
        
        // Get CUDA device for this rank
        int device = device_ids_[global_rank_];
        
        // Create ProcessGroup for this mesh dimension
        process_groups_[mesh_dim] = std::make_shared<ProcessGroup>(
            my_group_rank, group_size, device, nccl_id
        );
    }
}

ncclUniqueId DeviceMesh::create_nccl_id(int root_rank, MPI_Comm comm) {
    ncclUniqueId nccl_id;
    
    // Get our rank within this communicator
    int my_rank_in_comm;
    MPI_Comm_rank(comm, &my_rank_in_comm);
    
    if (my_rank_in_comm == root_rank) {
        ncclGetUniqueId(&nccl_id);
    }
    
    // Broadcast the NCCL ID within this sub-communicator only
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, root_rank, comm);
    
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
