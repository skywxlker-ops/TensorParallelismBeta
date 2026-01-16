#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "ProcessGroupNCCL.h"
#include "memory/cachingAllocator.hpp"


#include "bridge/bridge.h"
#include "device/Device.h"
#include "dtype/Dtype.h"


#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include "tensor/placement.h"


using namespace OwnTensor;


extern CachingAllocator gAllocator;

class DTensor {
public:

    DTensor(std::shared_ptr<DeviceMesh> device_mesh, std::shared_ptr<ProcessGroupNCCL> pg);
    
    /**
     * Constructor with layout - matches friend's API.
     * DTensor W1(device_mesh, pg, layout);
     */
    DTensor(DeviceMesh& device_mesh, std::shared_ptr<ProcessGroupNCCL> pg, const Layout& layout);
    
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
    // void sync_async();          // Queue sync (AllReduce SUM), don't wait

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
    DTensor relu() const;
    DTensor gelu() const;
    DTensor softmax(int64_t dim = -1) const;
    DTensor mse_loss(const DTensor& target) const;
    DTensor cross_entropy_loss(const DTensor& target) const;
    
    /**
     * Embedding lookup with autograd support.
     * @param indices Token IDs as OwnTensor (uint16 dtype)
     * @param weight Embedding weight DTensor [vocab_size, embedding_dim]
     * @param padding_idx Index to ignore (-1 for none)
     * @return DTensor of embeddings [num_tokens, embedding_dim]
     */
    static DTensor embedding(const OwnTensor::Tensor& indices, 
                             DTensor& weight, 
                             int padding_idx = -1);

    DTensor reshape(const std::vector<int64_t>& new_global_shape) const;

    /**
     * Redistribute tensor to a new layout.
     * Handles all transitions:
     *   - Partial → Replicate (AllReduce)
     *   - Partial → Shard (ReduceScatter)  
     *   - Shard → Replicate (AllGather)
     *   - Replicate → Shard (local slice)
     *   - Shard → Shard (AllGather + local slice)
     * @param target_layout Target layout to redistribute to
     * @return New DTensor with the target layout
     */
    DTensor redistribute(const Layout& target_layout) const;

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
    
    /**
     * Rotate a 3D tensor around the specified axis (in-place).
     * Also updates the shard dimension to follow the transpose.
     * @param dim Rotation axis (0=X, 1=Y, 2=Z)
     * @param direction Rotation direction (true=one way, false=opposite)
     */
    void rotate3D(int dim, bool direction);
    
    void saveCheckpoint(const std::string& path) const;
    void loadCheckpoint(const std::string& path);


    void print() const;
    
    /**
     * Display tensor values (debugging helper)
     */
    void display() const;
    
    /**
     * Initialize tensor with random values (in-place)
     */
    void rand();
    
    /**
     * Fused shard and transpose operation.
     * Shards source tensor along specified dimension and stores in this tensor.
     * @param shard_dim Dimension along which to shard the source
     * @param root Root rank for scatter operation
     * @param source Source DTensor to shard
     */
    void shard_fused_transpose(int shard_dim, int root, const DTensor& source);

    const Layout& get_layout() const { return layout_; }
    const OwnTensor::Tensor& local_tensor() const { return tensor_; }
    OwnTensor::Tensor& local_tensor() { return tensor_; }  // Non-const for weight updates
    std::shared_ptr<ProcessGroupNCCL> get_pg() const { return pg_; }
    std::shared_ptr<DeviceMesh> get_device_mesh() const { return device_mesh_; }
    int rank() const { return rank_; }

    // =========================================
    // AUTOGRAD INTERFACE
    // =========================================
    
    /**
     * Check if this DTensor requires gradient tracking.
     */
    bool requires_grad() const { return requires_grad_; }
    
    /**
     * Enable/disable gradient tracking.
     */
    void set_requires_grad(bool requires);
    
    /**
     * Get the gradient tensor (after backward).
     * Returns the local shard of the gradient.
     */
    OwnTensor::Tensor grad() const;
    
    /**
     * Compute gradients via backpropagation.
     * @param grad_output Initial gradient (default: ones like this tensor)
     */
    void backward(const DTensor* grad_output = nullptr);
    
    /**
     * Zero out gradients (for training loops).
     */
    void zero_grad();


    static DTensor empty(const std::vector<int64_t>& global_shape,
                         std::shared_ptr<DeviceMesh> mesh,
                         std::shared_ptr<ProcessGroupNCCL> pg,
                         const Layout& layout);
    
    /**
     * Create a DTensor filled with zeros.
     */
    static DTensor zeros(const std::vector<int64_t>& global_shape,
                         std::shared_ptr<DeviceMesh> mesh,
                         std::shared_ptr<ProcessGroupNCCL> pg,
                         const Layout& layout);
    
    /**
     * Create a DTensor filled with ones.
     */
    static DTensor ones(const std::vector<int64_t>& global_shape,
                        std::shared_ptr<DeviceMesh> mesh,
                        std::shared_ptr<ProcessGroupNCCL> pg,
                        const Layout& layout);
    
    /**
     * Create a DTensor filled with a constant value.
     */
    static DTensor full(const std::vector<int64_t>& global_shape,
                        float value,
                        std::shared_ptr<DeviceMesh> mesh,
                        std::shared_ptr<ProcessGroupNCCL> pg,
                        const Layout& layout);
    
    /**
     * Create a DTensor with uniform random values in [0, 1).
     */
    static DTensor rand(const std::vector<int64_t>& global_shape,
                        std::shared_ptr<DeviceMesh> mesh,
                        std::shared_ptr<ProcessGroupNCCL> pg,
                        const Layout& layout);
    
    /**
     * Create a DTensor with normal random values (mean=0, std=1).
     */
    static DTensor randn(const std::vector<int64_t>& global_shape,
                         std::shared_ptr<DeviceMesh> mesh,
                         std::shared_ptr<ProcessGroupNCCL> pg,
                         const Layout& layout);
    
    /**
     * Create a DTensor with random integers in [low, high).
     */
    static DTensor randint(int64_t low, int64_t high,
                           const std::vector<int64_t>& global_shape,
                           std::shared_ptr<DeviceMesh> mesh,
                           std::shared_ptr<ProcessGroupNCCL> pg,
                           const Layout& layout);
    
    /**
     * Create a DTensor from an existing local tensor shard.
     * This is useful when you already have the local data computed.
     */
    static DTensor from_local(const OwnTensor::Tensor& local_tensor,
                              std::shared_ptr<DeviceMesh> mesh,
                              std::shared_ptr<ProcessGroupNCCL> pg,
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
                                     std::shared_ptr<ProcessGroupNCCL> pg,
                                     const Layout& layout,
                                     int root = 0);

private:

    DTensor(std::shared_ptr<DeviceMesh> device_mesh,
            std::shared_ptr<ProcessGroupNCCL> pg,
            const OwnTensor::Tensor& local_tensor,
            const Layout& layout);


    DTensor _column_parallel_matmul(const DTensor& other) const;
    DTensor _row_parallel_matmul(const DTensor& other) const;

    // Helper for GPU-native initialization
    void _extract_local_shard(const OwnTensor::Tensor& full_tensor, const Layout& layout);
    
    // Lazy allocation helper - only reallocate temp_tensor_ if size changed
    void ensureTempTensor(const std::vector<int64_t>& shape);
    
    // Stream synchronization helpers for overlap
    void recordComputeDone();   // Record event on compute stream
    void recordCommDone();      // Record event on comm stream  
    void waitForCompute();      // Make comm stream wait for compute
    void waitForComm();         // Make compute stream wait for comm
    
    // Memory pool helper - get temporary buffer from caching allocator
    // void* getTempBuffer(size_t bytes);
    // void freeTempBuffer(void* ptr);


    int rank_;
    int world_size_;
    std::shared_ptr<DeviceMesh> device_mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    
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
    std::vector<int64_t> shape_; 
    std::string dtype_ = "float32";
    bool requires_grad_ = false;



    void printRecursive(const std::vector<float>& data,
                        const std::vector<int64_t>& dims,
                        int dim,
                        int offset) const;
};
