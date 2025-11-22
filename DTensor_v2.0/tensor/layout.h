#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <memory>
#include "tensor/mesh.h"

// Defines the sharding strategy for a DTensor
enum class ShardingType {
    REPLICATED, // Tensor is fully replicated on all devices
    SHARDED     // Tensor is sharded along a specific dimension
};

class Layout {
public:
    // Default constructor for an uninitialized/replicated layout
    Layout() : mesh_(nullptr), sharding_type_(ShardingType::REPLICATED), shard_dim_(-1) {}

    // Constructor for a new layout
    Layout(std::shared_ptr<Mesh> mesh, const std::vector<int>& global_shape, ShardingType type, int shard_dim = -1)
        : mesh_(mesh), global_shape_(global_shape), sharding_type_(type), shard_dim_(shard_dim) {
        
        if (sharding_type_ == ShardingType::SHARDED && (shard_dim < 0 || (size_t)shard_dim >= global_shape.size())) {
            throw std::runtime_error("Invalid shard_dim for SHARDED layout.");
        }
        if (sharding_type_ == ShardingType::REPLICATED) {
            shard_dim_ = -1; // Ensure shard_dim is -1 for replicated tensors
        }
    }

    // --- Accessors ---

    bool is_replicated() const {
        return sharding_type_ == ShardingType::REPLICATED;
    }

    bool is_sharded() const {
        return sharding_type_ == ShardingType::SHARDED;
    }

    // Check if sharded along a specific dimension
    bool is_sharded_by_dim(int dim) const {
        return is_sharded() && shard_dim_ == dim;
    }

    int get_shard_dim() const {
        return shard_dim_;
    }

    const std::vector<int>& get_global_shape() const {
        return global_shape_;
    }

    std::shared_ptr<Mesh> get_mesh() const {
        return mesh_;
    }

    // --- Core Logic ---

    // Calculates the shape of the local shard for a given rank
    std::vector<int> get_local_shape(int rank) const {
        if (is_replicated() || !mesh_ || global_shape_.empty()) {
            return global_shape_;
        }

        int world_size = mesh_->world_size;
        std::vector<int> local_shape = global_shape_;
        
        // This is the logic from your planner.h
        int global_dim_size = global_shape_[shard_dim_];
        int base_size = global_dim_size / world_size;
        int remainder = global_dim_size % world_size;
        
        int local_dim_size = (rank < remainder) ? (base_size + 1) : base_size;
        
        // Set the local shape for the sharded dimension
        local_shape[shard_dim_] = local_dim_size;
        
        return local_shape;
    }
    
    // Helper to get total number of elements in the global tensor
    int64_t global_numel() const {
        if (global_shape_.empty()) return 0;
        return std::accumulate(global_shape_.begin(), global_shape_.end(), 1LL, std::multiplies<int64_t>());
    }

    // Creates a new layout for a reshaped tensor (simplified)
    Layout reshape(const std::vector<int>& new_global_shape) const {
        // A simple check: if sharding dim is preserved and size is same, keep it.
        if (is_sharded() && shard_dim_ < (int)new_global_shape.size() && new_global_shape[shard_dim_] == global_shape_[shard_dim_]) {
             return Layout(mesh_, new_global_shape, ShardingType::SHARDED, shard_dim_);
        }
        
        // Default: fall back to replicated
        return Layout(mesh_, new_global_shape, ShardingType::REPLICATED);
    }

    // Check for element-wise op compatibility
    bool is_compatible(const Layout& other) const {
        // Simple check: are global shapes and sharding identical?
        return global_shape_ == other.global_shape_ &&
                sharding_type_ == other.sharding_type_ &&
                shard_dim_ == other.shard_dim_;
    }

    bool operator==(const Layout& other) const {
        return is_compatible(other);
    }

    bool operator!=(const Layout& other) const {
        return !is_compatible(other);
    }

    // --- Description Utility ---
    std::string describe(int rank) const {
        std::ostringstream oss;
        if (!mesh_) {
            oss << "[Layout] Uninitialized";
            return oss.str();
        }

        oss << "[Layout] Rank " << rank << "/" << mesh_->world_size << " | ";
        oss << "Global Shape: [";
        for (size_t i = 0; i < global_shape_.size(); ++i) {
            oss << global_shape_[i] << (i == global_shape_.size() - 1 ? "" : ", ");
        }
        oss << "] | ";

        if (is_replicated()) {
            oss << "REPLICATED";
        } else {
            oss << "SHARDED (Dim: " << shard_dim_ << ")";
        }
        
        std::vector<int> local_shape = get_local_shape(rank);
        oss << " | Local Shape: [";
        for (size_t i = 0; i < local_shape.size(); ++i) {
            oss << local_shape[i] << (i == local_shape.size() - 1 ? "" : ", ");
        }
        oss << "]";
        return oss.str();
    }


private:
    std::shared_ptr<Mesh> mesh_;
    std::vector<int> global_shape_;
    ShardingType sharding_type_;
    int shard_dim_; // Dimension along which tensor is sharded (-1 if REPLICATED)
};