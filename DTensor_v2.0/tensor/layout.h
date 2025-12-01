#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include "tensor/device_mesh.h"
#include "tensor/placement.h"


class Layout {
public:

    Layout() : device_mesh_(nullptr) {}
    
    Layout(std::shared_ptr<DeviceMesh> device_mesh, 
           const std::vector<int>& global_shape,
           const std::vector<std::shared_ptr<Placement>>& placements)
        : device_mesh_(device_mesh), global_shape_(global_shape), placements_(placements) {
        
        if (!device_mesh_) {
            throw std::runtime_error("Layout: device_mesh cannot be null");
        }
        
        if ((int)placements_.size() != device_mesh_->ndim()) {
            throw std::runtime_error("Layout: placements size must match mesh ndim");
        }
        
        validate_placements();
    }


    static Layout replicated(std::shared_ptr<DeviceMesh> device_mesh,
                            const std::vector<int>& global_shape) {
        std::vector<std::shared_ptr<Placement>> placements;
        for (int i = 0; i < device_mesh->ndim(); ++i) {
            placements.push_back(std::make_shared<Replicate>());
        }
        return Layout(device_mesh, global_shape, placements);
    }


    static Layout sharded_1d(std::shared_ptr<DeviceMesh> device_mesh,
                            const std::vector<int>& global_shape,
                            int shard_dim) {
        if (device_mesh->ndim() != 1) {
            throw std::runtime_error("sharded_1d requires 1D mesh");
        }
        std::vector<std::shared_ptr<Placement>> placements = {
            std::make_shared<Shard>(shard_dim)
        };
        return Layout(device_mesh, global_shape, placements);
    }

    const std::vector<int>& get_global_shape() const {
        return global_shape_;
    }

    std::shared_ptr<DeviceMesh> get_device_mesh() const {
        return device_mesh_;
    }

    const std::vector<std::shared_ptr<Placement>>& get_placements() const {
        return placements_;
    }

    std::shared_ptr<Placement> get_placement(int mesh_dim) const {
        if (mesh_dim < 0 || mesh_dim >= (int)placements_.size()) {
            throw std::runtime_error("Layout: invalid mesh_dim");
        }
        return placements_[mesh_dim];
    }


    bool is_fully_replicated() const {
        return std::all_of(placements_.begin(), placements_.end(),
            [](const auto& p) { return p->type() == PlacementType::REPLICATE; });
    }

    bool has_sharding() const {
        return std::any_of(placements_.begin(), placements_.end(),
            [](const auto& p) { return p->type() == PlacementType::SHARD; });
    }


    std::vector<int> get_sharded_dims() const {
        std::vector<int> dims;
        for (const auto& p : placements_) {
            if (p->type() == PlacementType::SHARD) {
                dims.push_back(static_cast<const Shard*>(p.get())->dim());
            }
        }
        return dims;
    }


    std::vector<int> get_local_shape(int rank) const {
        if (!device_mesh_ || global_shape_.empty()) {
            return global_shape_;
        }

        std::vector<int> local_shape = global_shape_;
        std::vector<int> mesh_coord = device_mesh_->get_coordinate(rank);


        for (int mesh_dim = 0; mesh_dim < device_mesh_->ndim(); ++mesh_dim) {
            auto placement = placements_[mesh_dim];
            
            if (placement->type() == PlacementType::SHARD) {
                int tensor_dim = static_cast<const Shard*>(placement.get())->dim();
                int mesh_size = device_mesh_->shape()[mesh_dim];
                int coord = mesh_coord[mesh_dim];
                
 
                int global_dim_size = local_shape[tensor_dim];
                int base_size = global_dim_size / mesh_size;
                int remainder = global_dim_size % mesh_size;
                
                int local_dim_size = (coord < remainder) ? (base_size + 1) : base_size;
                local_shape[tensor_dim] = local_dim_size;
            }
        
        }    
      

        return local_shape;
    }
    

    int64_t global_numel() const {
        if (global_shape_.empty()) return 0;
        return std::accumulate(global_shape_.begin(), global_shape_.end(), 
                              1LL, std::multiplies<int64_t>());
    }


    bool is_compatible(const Layout& other) const {
        if (global_shape_ != other.global_shape_) return false;
        if (placements_.size() != other.placements_.size()) return false;
        
        for (size_t i = 0; i < placements_.size(); ++i) {
            if (!placements_[i]->equals(other.placements_[i].get())) {
                return false;
            }
        }
        return true;
    }

    void row_parallel()


    std::string describe(int rank) const {
        std::ostringstream oss;
        if (!device_mesh_) {
            oss << "[Layout] Uninitialized";
            return oss.str();
        }

        oss << "[Layout] Rank " << rank << "/" << device_mesh_->world_size() << " | ";
 
        oss << "Global: [";
        for (size_t i = 0; i < global_shape_.size(); ++i) {
            oss << global_shape_[i];
            if (i < global_shape_.size() - 1) oss << ", ";
        }
        oss << "] | ";


        oss << "Placements: [";
        for (size_t i = 0; i < placements_.size(); ++i) {
            oss << placements_[i]->describe();
            if (i < placements_.size() - 1) oss << ", ";
        }
        oss << "]";
        

        std::vector<int> local_shape = get_local_shape(rank);
        oss << " | Local: [";
        for (size_t i = 0; i < local_shape.size(); ++i) {
            oss << local_shape[i];
            if (i < local_shape.size() - 1) oss << ", ";
        }
        oss << "]";
        
        return oss.str();
    }

private:
    std::shared_ptr<DeviceMesh> device_mesh_;
    std::vector<int> global_shape_;
    std::vector<std::shared_ptr<Placement>> placements_;  

    void validate_placements() {
        for (const auto& p : placements_) {
            if (p->type() == PlacementType::SHARD) {
                int dim = static_cast<const Shard*>(p.get())->dim();
                if (dim < 0 || dim >= (int)global_shape_.size()) {
                    throw std::runtime_error("Layout: Shard dim out of bounds");
                }
            }
        }
    }
};