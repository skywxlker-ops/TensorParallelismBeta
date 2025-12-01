#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "process_group/process_group.h"
#include "memory/cachingAllocator.hpp"

// --- Tensor & Ops Integration ---
#include "bridge/tensor_ops_bridge.h"
#include "device/Device.h"
#include "dtype/Dtype.h"

// --- NEW: Include Layout & Mesh ---
#include "tensor/device_mesh.h"
#include "tensor/layout.h"


using namespace OwnTensor;

// Global allocator declaration
extern CachingAllocator gAllocator;

class DTensor {
public:
    // === MODIFIED: Constructor now takes DeviceMesh and shared_ptr<ProcessGroup> ===
    DTensor(std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroup> pg);
    ~DTensor();

    // === Distributed collectives ===
    // These operate on the *local* tensor data
    void allReduce();
    void reduceScatter();
    void allGather();
    void broadcast(int root);

    // === MODIFIED: setData now requires a Layout ===
    // host_data must be the *local* data for this rank
    void setData(const std::vector<float>& host_data, const Layout& layout);
    
    // === GPU-Native Initialization ===
    // Load full tensor on root GPU, broadcast and scatter to other GPUs
    // host_data on root should contain the *full global* tensor
    // Non-root ranks can pass empty vector
    void setDataFromRoot(const std::vector<float>& host_data, const Layout& layout, int root = 0);
    
    std::vector<float> getData() const; // Gets local data

    // === Tensor Ops (now sharding-aware) ===
    DTensor add(const DTensor& other) const;
    DTensor sub(const DTensor& other) const;
    DTensor mul(const DTensor& other) const;
    DTensor div(const DTensor& other) const;
    DTensor matmul(const DTensor& other) const;

    // === View / Reshape (now layout-aware) ===
    DTensor reshape(const std::vector<int>& new_global_shape) const;
    DTensor redistribute(const Layout& new_layout) const;

    // === Checkpointing ===
    void saveCheckpoint(const std::string& path) const;
    void loadCheckpoint(const std::string& path);

    // === Debug utilities ===
    void print() const;
    
    // === Accessors ===
    const Layout& get_layout() const { return layout_; }
    const OwnTensor::Tensor& local_tensor() const { return tensor_; }
    std::shared_ptr<ProcessGroup> get_pg() const { return pg_; }
    std::shared_ptr<DeviceMesh> get_mesh() const { return mesh_; }
    int rank() const { return rank_; }


private:
    // === NEW: Private constructor for internal op results ===
    DTensor(std::shared_ptr<DeviceMesh> mesh,
            std::shared_ptr<ProcessGroup> pg,
            const OwnTensor::Tensor& local_tensor,
            const Layout& layout);

    // === NEW: Private matmul implementations ===
    DTensor _column_parallel_matmul(const DTensor& other) const;
    DTensor _row_parallel_matmul(const DTensor& other) const;
    
    // === NEW: Private helper for GPU-native initialization ===
    void _extract_local_shard(const OwnTensor::Tensor& full_tensor, const Layout& layout);

    // --- Core Members ---
    int rank_;
    int world_size_;
    std::shared_ptr<DeviceMesh> mesh_;
    std::shared_ptr<ProcessGroup> pg_;
    cudaStream_t stream_;

    Layout layout_; // Manages sharding, global shape, and local shape
    
    // --- Local Data ---
    OwnTensor::Tensor tensor_;      // This is the LOCAL shard
    OwnTensor::Tensor temp_tensor_; // Buffer for collectives

    // --- Legacy members (kept from your file for print/ckpt) ---
    int size_; // local size
    std::vector<int> shape_; // local shape
    std::string dtype_ = "float32";

    // --- Internal helpers ---
    void printRecursive(const std::vector<float>& data,
                        const std::vector<int>& dims,
                        int dim,
                        int offset) const;
};