#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <memory>
#include "tensor/device_mesh.h"
#include "tensor/placement.h"


class Layout {

    std::shared_ptr<Placement> placement_;
    std::vector<int64_t> global_shape_;

public: 
    // Default constructor for replicated layout
    Layout() : mesh_(nullptr), global_shape_({}), placement_(std::make_shared<Replicate>()) {}

    // Constructor for replicated layout with mesh and shape
    Layout(const DeviceMesh& mesh, const std::vector<int64_t> global_shape) 
        : mesh_(&mesh), global_shape_(global_shape), placement_(std::make_shared<Replicate>()) {}

    // Constructor for sharded layout
    Layout(const DeviceMesh& mesh, const std::vector<int64_t> global_shape, int dim)
        : mesh_(&mesh), global_shape_(global_shape), placement_(std::make_shared<Shard>(dim)) {
        if (placement_->dim() < 0 || (size_t)placement_->dim() >= global_shape.size()) {
            throw std::runtime_error("Invalid shard_dim for SHARDED layout.");
        }
    }

    // copy constructor
    Layout(const Layout& other) 
        : mesh_(other.mesh_), global_shape_(other.global_shape_),
          placement_(other.placement_ ? other.placement_->clone() : nullptr) {}

    // assignment operator
    Layout& operator=(const Layout& other) {
        if (this != &other) {
            mesh_ = other.mesh_;
            global_shape_ = other.global_shape_;
            placement_ = other.placement_ ? other.placement_->clone() : nullptr;
        }
        return *this;
    }

    // --- Accessors ---

    bool is_replicated() const {
        return placement_->type() == PlacementType::REPLICATE;
    }

    bool is_sharded() const {
        return placement_->type() == PlacementType::SHARD;
    }

    // Check if sharded along a specific dimension
    bool is_sharded_by_dim(int dim) const {
        return is_sharded() && placement_->dim() == dim;
    }

    int get_shard_dim() const {
        if (placement_->type() == PlacementType::SHARD) {
            return placement_->dim();
        }
        return -1; 
    }
    
    void set_shard_dim(int d) {
        placement_->setDim(d);       
    }

    const std::vector<int64_t>& get_global_shape() const {
        return global_shape_;
    }

    void set_global_shape(std::vector<int64_t> new_shape) {
        global_shape_ = new_shape;
    }

    const DeviceMesh& get_mesh() const {
        return *mesh_;
    }

    // --- Core Logic ---

    // Calculates the shape of the local shard for a given rank
    std::vector<int64_t> get_local_shape(int rank) const {
        if (placement_->type() == PlacementType::REPLICATE) {
            return global_shape_;
        }
    
        int shard_dim = get_shard_dim();
        if (shard_dim < 0 || (size_t)shard_dim >= global_shape_.size()) {
            return global_shape_;
        }

        int world_size = mesh_->world_size();
        if (world_size <= 0) return global_shape_;

        std::vector<int64_t> local_shape = global_shape_;
        int64_t global_dim_size = global_shape_[shard_dim];
        int64_t base_size = global_dim_size / world_size;
        int64_t remainder = global_dim_size % world_size;

        local_shape[shard_dim] = (rank < remainder) ? (base_size + 1) : base_size;
        return local_shape;
    }

    // Calculates the offset of the local shard for a given rank
    int64_t get_local_offset(int rank) const {
        if (placement_->type() == PlacementType::REPLICATE) {
            return 0;
        }

        int shard_dim = get_shard_dim();
        if (shard_dim < 0 || (size_t)shard_dim >= global_shape_.size()) {
            return 0;
        }

        int world_size = mesh_->world_size();
        if (world_size <= 0) return 0;

        int64_t global_dim_size = global_shape_[shard_dim];
        int64_t base_size = global_dim_size / world_size;
        int64_t remainder = global_dim_size % world_size;

        return (int64_t)rank * base_size + std::min((int64_t)rank, remainder);
    }


    // Helper to get total number of elements in the global tensor
    int64_t global_numel() const {
        if (global_shape_.empty()) return 0;
        return std::accumulate(global_shape_.begin(), global_shape_.end(), 1LL, std::multiplies<int64_t>());
    }

    // Creates a new layout for a reshaped tensor (simplified)
    Layout reshape(const std::vector<int64_t>& new_global_shape) const {
        // A simple check: if sharding dim is preserved and size is same, keep it.
        if (is_sharded() && placement_->dim() < (int64_t)new_global_shape.size() && 
            new_global_shape[placement_->dim()] == global_shape_[placement_->dim()]) {
             return Layout(*mesh_, new_global_shape, placement_->dim());
        }
        
        // Default: fall back to replicated
        return Layout(*mesh_, new_global_shape);
    }

    // Check for element-wise op compatibility with broadcasting
    bool is_compatible(const Layout& other) const {
        // Fast path: Exact match
        if (global_shape_ == other.global_shape_ &&
            placement_->type() == other.placement_->type() &&
            placement_->dim() == other.placement_->dim()) {
            return true;
        }

        // Broadcasting Check
        // Simplify for common case: 2D [B, N] + 1D [N]
        // LHS: [d0, d1], Sharded(1) -> local [d0, d1_local]
        // RHS: [d1],     Sharded(0) -> local [d1_local]
        // Result is valid if:
        // 1. Shapes are broadcast compatible globally
        // 2. Sharding maps to the same global dimension index
        
        // Basic global shape broadcast check (right-alignment)
        int ndim1 = global_shape_.size();
        int ndim2 = other.global_shape_.size();
        int max_dim = std::max(ndim1, ndim2);
        
        for (int i = 0; i < max_dim; ++i) {
            int idx1 = ndim1 - 1 - i;
            int idx2 = ndim2 - 1 - i;
            
            int64_t dim1 = (idx1 >= 0) ? global_shape_[idx1] : 1;
            int64_t dim2 = (idx2 >= 0) ? other.global_shape_[idx2] : 1;
            
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                return false; // Not broadcastable
            }
        }
        
        // Sharding compatibility
        // If both are replicated -> OK (handled by broadcast logic above implicitly? No, need to return true)
        if (is_replicated() && other.is_replicated()) return true;
        
        // If one is replicated and the other sharded:
        // Generally unsafe unless the sharded dim is 1 in the replicated tensor (so it stays 1 locally)
        // OR the replicated tensor effectively "broadcasts" over the sharded dimension.
        // For now, let's focus on the Sharded + Sharded case causing the crash.
        
        if (is_sharded() && other.is_sharded()) {
            int shard_dim1 = placement_->dim(); // Relative to ndim1
            int shard_dim2 = other.placement_->dim(); // Relative to ndim2
            
            // Map shard dimensions to the aligned "max_dim" space (right-aligned)
            int effective_dim1 = max_dim - (ndim1 - shard_dim1); 
            int effective_dim2 = max_dim - (ndim2 - shard_dim2);
            
            // They must be sharding the SAME global dimension
            if (effective_dim1 == effective_dim2) {
                // And that dimension must NOT be broadcasted (i.e. size must match)
                // (If one size was 1, it couldn't be sharded meaningfully in this context typically)
                return true;
            }
        }
        
        return false;
    }

    bool operator==(const Layout& other) const {
        return is_compatible(other);
    }

    bool operator!=(const Layout& other) const {
        return !is_compatible(other);
    }

    // --- Static Factory Methods ---
    
    static Layout replicated(const DeviceMesh& mesh, 
                            const std::vector<int64_t>& global_shape) {
        return Layout(mesh, global_shape);
    }

    // --- Description Utility ---
    std::string describe(int rank) const {
        std::ostringstream oss;

        oss << "[Layout] Rank " << rank << " | ";
        oss << "Global Shape: [";
        for (size_t i = 0; i < global_shape_.size(); ++i) {
            oss << global_shape_[i] << (i == global_shape_.size() - 1 ? "" : ", ");
        }
        oss << "] | ";

        if (is_replicated()) {
            oss << "REPLICATED";
        } else {
            oss << "SHARDED (Dim: " << placement_->dim() << ")";
        }
        
        std::vector<int64_t> local_shape = get_local_shape(rank);
        oss << " | Local Shape: [";
        for (size_t i = 0; i < local_shape.size(); ++i) {
            oss << local_shape[i] << (i == local_shape.size() - 1 ? "" : ", ");
        }
        oss << "]";
        return oss.str();
    }

    // --- Placement API ---
    
    std::shared_ptr<Placement> get_placement(int index = 0) const {
        (void)index; // Currently single placement
        return placement_;
    }

private:
    const DeviceMesh* mesh_;
};