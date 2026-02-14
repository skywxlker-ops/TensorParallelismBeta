#pragma once

/**
 * @file CustomDNN.h
 * @brief Customizable Distributed Neural Network Modules for Tensor Parallelism
 * 
 * This module provides a flexible, user-customizable framework for building
 * tensor-parallel neural networks on top of DTensor. Users have full control over:
 * - Device mesh dimensions and topology
 * - Parallelism strategies (column-parallel, row-parallel, replicated)
 * - Layer dimensions and configurations
 * - Communication patterns and NCCL process groups
 * 
 * The architecture is designed for experimentation and research in distributed
 * deep learning, allowing easy customization of all parallelism parameters.
 */

#include "TensorLib.h"
#include <memory>
#include <vector>

// Forward declarations
class DTensor;
class DeviceMesh;
class ProcessGroupNCCL;
class Layout;

namespace OwnTensor {
namespace dnn {

// =============================================================================
// Sharding Configuration
// =============================================================================

/**
 * @class ShardingType
 * @brief Specifies how a tensor should be distributed across devices
 * 
 * Usage:
 *   ShardingType::Shard(0)    - Shard along dimension 0 (row-parallel for weight)
 *   ShardingType::Shard(1)    - Shard along dimension 1 (column-parallel for weight)
 *   ShardingType::Replicated() - Full copy on each device
 */
class ShardingType {
public:
    enum class Type { SHARD, REPLICATED };
    
    /**
     * @brief Create a sharding type that splits tensor along specified dimension
     * @param dim Dimension to shard (0 = rows, 1 = columns)
     */
    static ShardingType Shard(int dim) {
        ShardingType st;
        st.type_ = Type::SHARD;
        st.dim_ = dim;
        return st;
    }
    
    /**
     * @brief Create a replicated sharding (full copy on each device)
     */
    static ShardingType Replicated() {
        ShardingType st;
        st.type_ = Type::REPLICATED;
        st.dim_ = -1;
        return st;
    }
    
    bool is_shard() const { return type_ == Type::SHARD; }
    bool is_replicated() const { return type_ == Type::REPLICATED; }
    int shard_dim() const { return dim_; }
    
private:
    Type type_ = Type::REPLICATED;
    int dim_ = -1;
};

// Legacy enum for backward compatibility
enum class ParallelType {
    COLUMN,  ///< Shard output dimension (column-wise) = Shard(1)
    ROW      ///< Shard input dimension (row-wise) = Shard(0)
};

// =============================================================================
// Base Distributed Module
// =============================================================================

/**
 * @class DModule
 * @brief Base class for distributed neural network modules
 */
class DModule {
public:
    virtual ~DModule() = default;
    
    /**
     * @brief Set gradient requirement for all parameters
     */
    virtual void set_requires_grad(bool requires) = 0;
    
    /**
     * @brief Zero all parameter gradients
     */
    virtual void zero_grad() = 0;
    
    /**
     * @brief Get all trainable parameters
     */
    virtual std::vector<DTensor*> parameters() = 0;

protected:
    std::shared_ptr<DeviceMesh> mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
};

// =============================================================================
// Distributed Linear Layer
// =============================================================================

/**
 * @class DLinear
 * @brief Tensor-parallel linear layer with flexible sharding
 * 
 * Implements a distributed linear transformation Y = XW + b with support for
 * per-parameter sharding strategies.
 * 
 * Auto-sync behavior:
 * - If weight is Shard(0) (row-parallel), forward() automatically does AllReduce
 * - If weight is Shard(1) (column-parallel), no sync needed in forward()
 * 
 * Example usage:
 * @code
 *   // Column-parallel: shard weight columns, replicate bias
 *   auto fc1 = DLinear(mesh, pg, 768, 3072,
 *                      ShardingType::Shard(1),       // weight: column-wise
 *                      ShardingType::Replicated(),   // bias: replicated
 *                      true);                        // has_bias
 *   
 *   // Row-parallel: shard weight rows, no bias
 *   auto fc2 = DLinear(mesh, pg, 3072, 768,
 *                      ShardingType::Shard(0),       // weight: row-wise
 *                      ShardingType::Replicated(),   // bias: replicated
 *                      true);                        // has_bias
 * @endcode
 */
class DLinear : public DModule {
public:
    /**
     * @brief NEW API: Construct with explicit sharding for weight and bias
     * @param mesh Device mesh for tensor distribution
     * @param pg Process group for collective communication
     * @param in_features Input dimension
     * @param out_features Output dimension
     * @param weight_sharding How to shard the weight matrix
     * @param bias_sharding How to shard the bias vector
     * @param has_bias Whether to include a bias term
     */
    DLinear(std::shared_ptr<DeviceMesh> mesh,
            std::shared_ptr<ProcessGroupNCCL> pg,
            int64_t in_features,
            int64_t out_features,
            ShardingType weight_sharding,
            ShardingType bias_sharding = ShardingType::Replicated(),
            bool has_bias = true);
    
    /**
     * @brief LEGACY: Construct with ParallelType (backward compatible)
     */
    DLinear(int64_t in_features,
            int64_t out_features,
            std::shared_ptr<DeviceMesh> mesh,
            std::shared_ptr<ProcessGroupNCCL> pg,
            ParallelType parallel_type);
    
    /**
     * @brief Forward pass with auto-sync
     * 
     * Automatically handles synchronization:
     * - Row-parallel (Shard(0)) -> AllReduce after matmul
     * - Column-parallel (Shard(1)) -> No sync needed
     * 
     * @param input Input DTensor
     * @param no_sync If true, skip internal AllReduce (caller will sync later)
     * @return Output DTensor after linear transformation
     */
    DTensor forward(const DTensor& input, bool no_sync = false);
    
    /**
     * @brief Get weight tensor
     */
    DTensor& weight();
    
    /**
     * @brief Get bias tensor (throws if has_bias=false)
     */
    DTensor& bias();
    
    /**
     * @brief Check if layer has bias
     */
    bool has_bias() const { return has_bias_; }
    
    void set_requires_grad(bool requires) override;
    void zero_grad() override;
    std::vector<DTensor*> parameters() override;

private:
    int64_t in_features_;
    int64_t out_features_;
    bool has_bias_;
    ShardingType weight_sharding_;
    ShardingType bias_sharding_;
    ParallelType parallel_type_;  // For legacy compatibility
    std::unique_ptr<DTensor> weight_;  ///< [in_features, out_features] (sharded)
    std::unique_ptr<DTensor> bias_;    ///< [out_features] (sharded or replicated)
};

// =============================================================================
// Distributed Replicated Linear Layer
// =============================================================================

/**
 * @class DLinearReplicated
 * @brief Non-parallel (replicated) linear layer
 * 
 * Each GPU holds the full weight matrix. Useful for output projections
 * where parallelism is not desired.
 */
class DLinearReplicated : public DModule {
public:
    /**
     * @brief Construct a replicated linear layer
     * @param in_features Input dimension
     * @param out_features Output dimension
     * @param mesh Device mesh
     * @param pg Process group
     */
    DLinearReplicated(int64_t in_features,
                      int64_t out_features,
                      std::shared_ptr<DeviceMesh> mesh,
                      std::shared_ptr<ProcessGroupNCCL> pg);
    
    DTensor forward(const DTensor& input);
    
    DTensor& weight();
    void set_requires_grad(bool requires) override;
    void zero_grad() override;
    std::vector<DTensor*> parameters() override;

private:
    int64_t in_features_;
    int64_t out_features_;
    std::unique_ptr<DTensor> weight_;  ///< [in_features, out_features] replicated
};

// =============================================================================
// Distributed MLP
// =============================================================================

/**
 * @class DMLP
 * @brief Two-layer tensor-parallel MLP
 * 
 * Architecture:
 * - fc1: Column-parallel linear (expands to hidden_features)
 * - activation: GeLU
 * - fc2: Row-parallel linear (reduces to out_features)
 */
class DMLP : public DModule {
public:
    /**
     * @brief Construct a distributed MLP
     * @param in_features Input dimension
     * @param hidden_features Hidden dimension
     * @param out_features Output dimension
     * @param mesh Device mesh
     * @param pg Process group
     */
    DMLP(int64_t in_features,
         int64_t hidden_features,
         int64_t out_features,
         std::shared_ptr<DeviceMesh> mesh,
         std::shared_ptr<ProcessGroupNCCL> pg);
    
    /**
     * @brief Forward pass
     * @param input Input DTensor
     * @return Output DTensor
     */
    DTensor forward(const DTensor& input);
    
    void set_requires_grad(bool requires) override;
    void zero_grad() override;
    std::vector<DTensor*> parameters() override;
    
    // Layer accessors
    DLinear& fc1();
    DLinear& fc2();

private:
    std::unique_ptr<DLinear> fc1_;  ///< First layer (column-parallel)
    std::unique_ptr<DLinear> fc2_;  ///< Second layer (row-parallel)
};

// =============================================================================
// Distributed Embedding
// =============================================================================

/**
 * @class DEmbedding
 * @brief Distributed embedding layer with optional vocab sharding
 * 
 * Supports two modes:
 * - Replicated: Full embedding table on each GPU (default, backward compat)
 * - Sharded (Row Parallel): Vocab sharded across GPUs, uses AllReduce in forward
 * 
 * Row Parallel sharding reduces memory: each GPU stores vocab_size/world_size rows.
 */
class DEmbedding : public DModule {
public:
    /**
     * @brief Construct a distributed embedding layer
     * @param vocab_size Global vocabulary size
     * @param embedding_dim Embedding dimension
     * @param mesh Device mesh
     * @param pg Process group
     * @param sharding Sharding type (Replicated or Shard(0) for vocab sharding)
     */
    DEmbedding(int64_t vocab_size,
               int64_t embedding_dim,
               std::shared_ptr<DeviceMesh> mesh,
               std::shared_ptr<ProcessGroupNCCL> pg,
               ShardingType sharding = ShardingType::Replicated());
    
    /**
     * @brief Forward pass: lookup embeddings
     * @param token_ids Vector of token IDs [batch_size * seq_len]
     * @return DTensor of embeddings [batch_size * seq_len, embedding_dim]
     * 
     * If sharded, performs local lookup + AllReduce to combine partial results.
     */
    DTensor forward(const std::vector<int32_t>& token_ids);

    /**
     * @brief Forward pass: lookup embeddings using DTensor indices [B, T]
     */
    DTensor forward(const DTensor& indices);
    
    DTensor& weight();
    void set_requires_grad(bool requires) override;
    void zero_grad() override;
    std::vector<DTensor*> parameters() override;
    
    // Accessors
    int64_t vocab_size() const { return vocab_size_; }
    int64_t local_vocab_size() const { return local_vocab_size_; }
    int64_t embedding_dim() const { return embedding_dim_; }
    bool is_sharded() const { return sharding_.is_shard(); }

private:
    int64_t vocab_size_;        ///< Global vocab size
    int64_t local_vocab_size_;  ///< Local vocab size (= global / world_size if sharded)
    int64_t embedding_dim_;
    int64_t vocab_start_idx_;   ///< Start index of local vocab range
    ShardingType sharding_;
    std::unique_ptr<DTensor> weight_;  ///< [local_vocab_size, embedding_dim]
};

// =============================================================================
// Distributed Cross Entropy Loss
// =============================================================================

/**
 * @class CrossEntropyLoss
 * @brief Distributed Cross Entropy Loss module
 * 
 * Computes categorical cross entropy between logits and targets.
 * Supports DTensor inputs and handles gradient calculation internally.
 */
class CrossEntropyLoss : public DModule {
public:
    CrossEntropyLoss(std::shared_ptr<DeviceMesh> mesh,
                     std::shared_ptr<ProcessGroupNCCL> pg);
    
    /**
     * @brief Forward pass: compute loss
     * @param logits Prediction logits [B, Vocab]
     * @param targets Target distributions or indices [B, Vocab] (one-hot)
     * @return Scalar DTensor containing the loss
     */
    DTensor forward(const DTensor& logits, const DTensor& targets);

    void set_requires_grad(bool requires) override {}
    void zero_grad() override {}
    std::vector<DTensor*> parameters() override { return {}; }
};

// =============================================================================
// Distributed Layer Normalization
// =============================================================================

/**
 * @class DLayerNorm
 * @brief Distributed Layer Normalization
 */
class DLayerNorm : public DModule {
public:
    DLayerNorm(int normalized_shape,
               std::shared_ptr<DeviceMesh> mesh,
               std::shared_ptr<ProcessGroupNCCL> pg,
               float eps = 1e-5);
    
    DTensor forward(const DTensor& input);
    
    void set_requires_grad(bool requires) override;
    void zero_grad() override;
    std::vector<DTensor*> parameters() override;

private:
    int normalized_shape_;
    float eps_;
    std::unique_ptr<DTensor> weight_; ///< [normalized_shape] replicated
    std::unique_ptr<DTensor> bias_;   ///< [normalized_shape] replicated
};

// =============================================================================
// Optimizers
// =============================================================================

/**
 * @class SGD
 * @brief Simple Stochastic Gradient Descent optimizer
 */
class SGD {
public:
    explicit SGD(float lr = 0.01f) : lr_(lr) {}
    void step(std::vector<DTensor*> params);
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }
private:
    float lr_;
};

/**
 * @class AdamW
 * @brief AdamW optimizer with gradient clipping and weight decay
 */
class AdamW {
public:
    /**
     * @brief Construct AdamW optimizer
     * @param lr Learning rate
     * @param beta1 First moment decay
     * @param beta2 Second moment decay
     * @param eps Epsilon for numerical stability
     * @param weight_decay Weight decay (L2 penalty)
     */
    explicit AdamW(float lr, float beta1 = 0.9f, float beta2 = 0.999f, 
                   float eps = 1e-8f, float weight_decay = 0.01f)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), 
          weight_decay_(weight_decay), t_(0) {}
    
    /**
     * @brief Perform optimization step with AdamW logic
     * @param params Vector of parameter tensors
     */
    void step(std::vector<DTensor*> params);
    
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }
    

private:
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    int t_;
    // State: first and second moments for each parameter
    // Keys are DTensor pointers, values are local Tensor shards on GPU
    std::unordered_map<DTensor*, OwnTensor::Tensor> m_;
    std::unordered_map<DTensor*, OwnTensor::Tensor> v_;
};


} // namespace dnn
} // namespace OwnTensor
