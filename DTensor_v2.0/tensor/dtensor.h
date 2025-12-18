#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "process_group.h"
#include "memory/cachingAllocator.hpp"


#include "bridge/tensor_ops_bridge.h"
#include "device/Device.h"
#include "dtype/Dtype.h"


#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include "tensor/placement.h"
#include "tensor/activation_kernels.h"


using namespace OwnTensor;


extern CachingAllocator gAllocator;

class DTensor {
public:

    DTensor(std::shared_ptr<DeviceMesh> device_mesh, std::shared_ptr<ProcessGroup> pg);
    ~DTensor();

    // Collective Communication Operations
    void allReduce();           // In-place all-reduce (blocking)
    void reduceScatter();       // Reduce and scatter (blocking)
    void allGather();           // All-gather operation (blocking)
    void broadcast(int root);   // Broadcast from root (blocking)
    void scatter(int root);     // Scatter from root (blocking)
    void sync();                // AllReduce with SUM (blocking with overlap)
    
    // Async variants for use within NCCL groups (non-blocking)
    void allReduce_async();     // Queue AllReduce, don't wait
    void reduceScatter_async(); // Queue ReduceScatter, don't wait
    void allGather_async();     // Queue AllGather, don't wait
    void sync_async();          // Queue sync (AllReduce SUM), don't wait

    void setData(const std::vector<float>& host_data, const Layout& layout);
    
    // GPU-Native Initialization
    // Load full tensor on root GPU, broadcast and scatter to other GPUs
    // host_data on root should contain the *full global* tensor
    // Non-root ranks can pass empty vector
    void setDataFromRoot(const std::vector<float>& host_data, const Layout& layout, int root = 0);
    
    std::vector<float> getData() const; 

    DTensor add(const DTensor& other) const;
    DTensor sub(const DTensor& other) const;
    DTensor mul(const DTensor& other) const;
    DTensor div(const DTensor& other) const;
    DTensor matmul(const DTensor& other) const;
    
    // Fused matmul + activation operations for kernel fusion optimization
    DTensor matmul_gelu(const DTensor& other) const;
    DTensor matmul_relu(const DTensor& other) const;

    DTensor reshape(const std::vector<int>& new_global_shape) const;

    // Layout transformations (in-place)
    /**
     * Replicate tensor to all devices using Broadcast (in-place).
     * Broadcasts tensor from root GPU to all other GPUs.
     * Modifies this tensor to have replicated layout.
     * @param root Root rank that has the data to broadcast (default: 0)
     */
    void replicate(int root = 0);
    
    /**
     * Shard tensor across devices (in-place).
     * Distributes tensor along specified dimension to all GPUs.
     * Can be called on any tensor (not just replicated ones).
     * @param dim Dimension along which to shard
     * @param root Root rank for initial data distribution (default: 0)
     */
    void shard(int dim, int root = 0);


    /**
     * Scale tensor values by a factor (in-place).
     * Multiplies all elements by the given scalar.
     */
    void scale(float factor);
    void saveCheckpoint(const std::string& path) const;
    void loadCheckpoint(const std::string& path);


    void print() const;
    

    const Layout& get_layout() const { return layout_; }
    const OwnTensor::Tensor& local_tensor() const { return tensor_; }
    std::shared_ptr<ProcessGroup> get_pg() const { return pg_; }
    std::shared_ptr<DeviceMesh> get_device_mesh() const { return device_mesh_; }
    int rank() const { return rank_; }

    // =========================================================================
    // Static Factory Functions (PyTorch-style)
    // =========================================================================
    // These create DTensors directly with sharded memory allocation,
    // without ever needing a global tensor.
    
    /**
     * Create an uninitialized DTensor (fastest - no memset).
     */
    static DTensor empty(const std::vector<int>& global_shape,
                         std::shared_ptr<DeviceMesh> mesh,
                         std::shared_ptr<ProcessGroup> pg,
                         const Layout& layout);
    
    /**
     * Create a DTensor filled with zeros.
     */
    static DTensor zeros(const std::vector<int>& global_shape,
                         std::shared_ptr<DeviceMesh> mesh,
                         std::shared_ptr<ProcessGroup> pg,
                         const Layout& layout);
    
    /**
     * Create a DTensor filled with ones.
     */
    static DTensor ones(const std::vector<int>& global_shape,
                        std::shared_ptr<DeviceMesh> mesh,
                        std::shared_ptr<ProcessGroup> pg,
                        const Layout& layout);
    
    /**
     * Create a DTensor filled with a constant value.
     */
    static DTensor full(const std::vector<int>& global_shape,
                        float value,
                        std::shared_ptr<DeviceMesh> mesh,
                        std::shared_ptr<ProcessGroup> pg,
                        const Layout& layout);
    
    /**
     * Create a DTensor with uniform random values in [0, 1).
     */
    static DTensor rand(const std::vector<int>& global_shape,
                        std::shared_ptr<DeviceMesh> mesh,
                        std::shared_ptr<ProcessGroup> pg,
                        const Layout& layout);
    
    /**
     * Create a DTensor with normal random values (mean=0, std=1).
     */
    static DTensor randn(const std::vector<int>& global_shape,
                         std::shared_ptr<DeviceMesh> mesh,
                         std::shared_ptr<ProcessGroup> pg,
                         const Layout& layout);
    
    /**
     * Create a DTensor with random integers in [low, high).
     */
    static DTensor randint(int64_t low, int64_t high,
                           const std::vector<int>& global_shape,
                           std::shared_ptr<DeviceMesh> mesh,
                           std::shared_ptr<ProcessGroup> pg,
                           const Layout& layout);
    
    /**
     * Create a DTensor from an existing local tensor shard.
     * This is useful when you already have the local data computed.
     */
    static DTensor from_local(const OwnTensor::Tensor& local_tensor,
                              std::shared_ptr<DeviceMesh> mesh,
                              std::shared_ptr<ProcessGroup> pg,
                              const Layout& layout);

    /**
     * Distribute an existing global tensor to create a DTensor (GPU-native).
     * The tensor should exist on the root GPU, and will be scattered/broadcast
     * according to the layout. Non-root ranks can pass any tensor (ignored).
     * 
     * This is the GPU-native equivalent of setDataFromRoot().
     * 
     * @param global_tensor The full tensor on root GPU
     * @param mesh Device mesh for distribution
     * @param pg Process group for NCCL communication
     * @param layout Target layout (replicated or sharded)
     * @param root Root rank that has the global tensor (default: 0)
     */
    static DTensor distribute_tensor(const OwnTensor::Tensor& global_tensor,
                                     std::shared_ptr<DeviceMesh> mesh,
                                     std::shared_ptr<ProcessGroup> pg,
                                     const Layout& layout,
                                     int root = 0);

private:

    DTensor(std::shared_ptr<DeviceMesh> device_mesh,
            std::shared_ptr<ProcessGroup> pg,
            const OwnTensor::Tensor& local_tensor,
            const Layout& layout);


    DTensor _column_parallel_matmul(const DTensor& other) const;
    DTensor _row_parallel_matmul(const DTensor& other) const;

    // Helper for GPU-native initialization
    void _extract_local_shard(const OwnTensor::Tensor& full_tensor, const Layout& layout);
    
    // Lazy allocation helper - only reallocate temp_tensor_ if size changed
    void ensureTempTensor(const std::vector<int>& shape);
    
    // Stream synchronization helpers for overlap
    void recordComputeDone();   // Record event on compute stream
    void recordCommDone();      // Record event on comm stream  
    void waitForCompute();      // Make comm stream wait for compute
    void waitForComm();         // Make compute stream wait for comm
    
    // Memory pool helper - get temporary buffer from caching allocator
    void* getTempBuffer(size_t bytes);
    void freeTempBuffer(void* ptr);


    int rank_;
    int world_size_;
    std::shared_ptr<DeviceMesh> device_mesh_;
    std::shared_ptr<ProcessGroup> pg_;
    
    // Multi-stream support for concurrent execution
    cudaStream_t compute_stream_;  // For matmul, activations
    cudaStream_t comm_stream_;     // For NCCL collectives
    cudaStream_t data_stream_;     // For data transfers
    
    // Events for stream synchronization (enable overlap)
    cudaEvent_t compute_event_;    // Mark compute operations complete
    cudaEvent_t comm_event_;       // Mark communication complete

    Layout layout_;
    
    OwnTensor::Tensor tensor_;      
    OwnTensor::Tensor temp_tensor_;  // TEMPORARY: Restored for debugging

    int size_;
    std::vector<int> shape_; 
    std::string dtype_ = "float32";

    
    // Fused operations (Phase 4 optimization)
    DTensor matmul_bias_gelu(const DTensor& weights, const DTensor& bias) const;
    DTensor matmul_bias_relu(const DTensor& weights, const DTensor& bias) const;
    void add_bias(const DTensor& bias);

    void printRecursive(const std::vector<float>& data,
                        const std::vector<int>& dims,
                        int dim,
                        int offset) const;
};
