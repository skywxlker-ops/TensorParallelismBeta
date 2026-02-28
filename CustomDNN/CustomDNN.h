#pragma once

/**
 * @file CustomDNN.h
 * @brief Customizable Distributed Neural Network Modules for Tensor Parallelism
 * 
 * Built on top of DTensor, this module provides a flexible, user-customizable
 * framework for building tensor-parallel neural networks. Users have full
 * control over:
 * - Parallelism strategies via ShardingType (column-parallel, row-parallel, replicated)
 * - Layer dimensions and configurations
 * - Communication patterns
 * 
 * Unlike DTensor's built-in DistributedNN.h (which hardcodes parallelism per class),
 * CustomDNN lets users specify ShardingType per parameter for maximum flexibility.
 * 
 * NOTE: This header does NOT include DistributedNN.h to avoid multiple-definition
 * linker errors (DistributedNN.h has non-inline function definitions). Instead,
 * we define our own DModuleBase class with the same API.
 */

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "ops/helpers/MultiTensorKernels.h"
#include "nn/NN.h"
#include "tensor/dtensor.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include "process_group/ProcessGroupNCCL.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace CustomDNN {

// =============================================================================
// DModuleBase — mirrors DTensor's dnn::DModule but defined here to avoid
// including DistributedNN.h (which has non-inline free function definitions)
// =============================================================================

class DModuleBase {
public:
    virtual ~DModuleBase() {
        for (auto* p : params_) delete p;
    }
    DModuleBase() = default;
    DModuleBase(const DModuleBase&) = default;
    DModuleBase& operator=(const DModuleBase&) = default;
    DModuleBase(DModuleBase&&) = default;
    DModuleBase& operator=(DModuleBase&&) = default;

    virtual DTensor forward(DTensor& input) {
        throw std::runtime_error("DModuleBase::forward not implemented");
    }

    virtual std::vector<DTensor*> parameters() {
        std::vector<DTensor*> all_params = params_;
        for (auto* child : children_) {
            auto child_params = child->parameters();
            all_params.insert(all_params.end(), child_params.begin(), child_params.end());
        }
        return all_params;
    }

    virtual void to(OwnTensor::DeviceIndex dev) {
        for (auto* child : children_) {
            child->to(dev);
        }
    }

    virtual void all_reduce_gradients(ProcessGroupNCCL* pg) {
        for (auto* child : children_) {
            child->all_reduce_gradients(pg);
        }
    }

    void zero_grad() {
        for (DTensor* p : parameters()) {
            if (p && p->mutable_tensor().requires_grad() && p->mutable_tensor().has_grad()) {
                p->mutable_tensor().zero_grad();
            }
        }
    }

protected:
    std::vector<DTensor*> params_;
    std::vector<DModuleBase*> children_;

    void register_parameter(OwnTensor::Tensor* p) {
        DTensor* tensor = new DTensor();
        tensor->set_tensor(*p);
        // Default to a replicated layout using the tensor's own shape
        tensor->setShape(p->shape().dims); 
        params_.push_back(tensor);
    }



    void register_module(DModuleBase& m) {
        children_.push_back(&m);
    }

    void register_module(DModuleBase* m) {
        children_.push_back(m);
    }
};

// =============================================================================
// DSequential — container for sequential modules
// =============================================================================

class DSequential : public DModuleBase {
public:
    void add(std::shared_ptr<DModuleBase> module) {
        modules_.push_back(module);
        register_module(*module);
    }

    DTensor forward(DTensor& input) override {
        DTensor x = input;
        for (auto& m : modules_) {
            x = m->forward(x);
        }
        return x;
    }

    void all_reduce_gradients(ProcessGroupNCCL* pg) override {
        for (auto& m : modules_) {
            m->all_reduce_gradients(pg);
        }
    }

    auto& operator[](size_t idx) { return *modules_[idx]; }
    size_t size() const { return modules_.size(); }

private:
    std::vector<std::shared_ptr<DModuleBase>> modules_;
};

// =============================================================================
// DGeLU — GeLU activation for DTensors
// =============================================================================

class DGeLU : public DModuleBase {
public:
    DTensor forward(DTensor& input) override {
        OwnTensor::Tensor in_tensor = input.mutable_tensor();
        OwnTensor::Tensor out_tensor = OwnTensor::autograd::gelu(in_tensor);
        DTensor output(input.get_device_mesh(), input.get_pg(), input.get_layout());
        output.mutable_tensor() = out_tensor;
        return output;
    }
};

// =============================================================================
// Gradient Clipping for DTensor params (standalone, avoids DistributedNN.h)
// =============================================================================

float clip_grad_norm_dtensor_nccl(
    std::vector<DTensor*>& params,
    float max_norm,
    std::shared_ptr<ProcessGroupNCCL> pg,
    float norm_type = 2.0f);

// =============================================================================
// Sharding Configuration
// =============================================================================

/**
 * @class ShardingType
 * @brief Specifies how a tensor should be distributed across devices
 * 
 * Usage:
 *   ShardingType::Shard(0)     - Shard along dimension 0 (row-parallel for weight)
 *   ShardingType::Shard(1)     - Shard along dimension 1 (column-parallel for weight)
 *   ShardingType::Replicated() - Full copy on each device
 */
class ShardingType {
public:
    enum class Type { SHARD, REPLICATED };
    
    static ShardingType Shard(int dim) {
        ShardingType st;
        st.type_ = Type::SHARD;
        st.dim_ = dim;
        return st;
    }
    
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

// =============================================================================
// Distributed Linear Layer
// =============================================================================

/**
 * @class DLinear
 * @brief Tensor-parallel linear layer with flexible sharding
 * 
 * Auto-sync behavior:
 * - If weight is Shard(0) (row-parallel), forward() automatically does AllReduce
 * - If weight is Shard(1) (column-parallel), no sync needed in forward()
 */
class DLinear : public DModuleBase {
public:
    DLinear(const DeviceMesh& mesh,
            std::shared_ptr<ProcessGroupNCCL> pg,
            int64_t batch_size,
            int64_t seq_len,
            int64_t in_features,
            int64_t out_features,
            ShardingType weight_sharding,
            bool has_bias = false,
            float sd = 0.02f,
            int seed = 42);
    
    DTensor forward(DTensor& input) override;
    
    DTensor& weight() { return *weight_; }
    DTensor& bias() { 
        if (!has_bias_ || !bias_) throw std::runtime_error("DLinear: no bias");
        return *bias_; 
    }
    bool has_bias() const { return has_bias_; }
    ShardingType get_sharding() const { return weight_sharding_; }

    void all_reduce_gradients(ProcessGroupNCCL* pg) override;

private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    int64_t in_features_;
    int64_t out_features_;
    int64_t out_local_;
    int64_t batch_size_;
    int64_t seq_len_;
    bool has_bias_;
    ShardingType weight_sharding_;
    bool is_row_parallel_;
    std::unique_ptr<DTensor> weight_;
    std::unique_ptr<DTensor> bias_;
};

// =============================================================================
// Distributed MLP
// =============================================================================

/**
 * @class DMLP
 * @brief Configurable two-layer tensor-parallel MLP
 * 
 * Architecture: fc1 (column-parallel) → GeLU → fc2 (row-parallel, auto-sync)
 */
class DMLP : public DModuleBase {
public:
    DMLP(const DeviceMesh& mesh,
         std::shared_ptr<ProcessGroupNCCL> pg,
         int64_t batch_size,
         int64_t seq_len,
         int64_t in_features,
         int64_t hidden_features,
         int64_t out_features,
         bool has_bias = false,
         float residual_scale = 1.0f,
         int seed = 42);
    
    DTensor forward(DTensor& input) override;
    void all_reduce_gradients(ProcessGroupNCCL* pg) override;
    
    DLinear& fc1() { return *fc1_; }
    DLinear& fc2() { return *fc2_; }

private:
    std::unique_ptr<DLinear> fc1_;
    std::unique_ptr<DLinear> fc2_;
    DGeLU gelu_;
};

// =============================================================================
// Distributed Block
// =============================================================================

/**
 * @class DBlock
 * @brief Matches the exact architecture of gpt2_tp_test's MLP block.
 * 
 * Architecture: pre-LN → DMLP → Residual Add
 */
class DLayerNorm;

class DBlock : public DModuleBase {
public:
    DBlock(const DeviceMesh& mesh,
         std::shared_ptr<ProcessGroupNCCL> pg,
         int64_t batch_size,
         int64_t seq_len,
         int64_t n_embd,
         int n_layers,
         int seed = 42);
    
    DTensor forward(DTensor& input) override;
    void all_reduce_gradients(ProcessGroupNCCL* pg) override;

private:
    std::unique_ptr<DLayerNorm> ln_;
    std::unique_ptr<DMLP> mlp_;
};

// =============================================================================
// Distributed Embedding
// =============================================================================

class DEmbedding : public DModuleBase {
public:
    DEmbedding(const DeviceMesh& mesh,
               std::shared_ptr<ProcessGroupNCCL> pg,
               int64_t vocab_size,
               int64_t embedding_dim,
               ShardingType sharding = ShardingType::Replicated(),
               float sd = 0.02f,
               int seed = 42);
    
    DTensor forward(DTensor& input) override;
    
    DTensor& weight() { return *weight_; }
    int64_t vocab_size() const { return vocab_size_; }
    int64_t embedding_dim() const { return embedding_dim_; }
    void all_reduce_gradients(ProcessGroupNCCL* pg) override;

private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    int64_t vocab_size_;
    int64_t embedding_dim_;
    ShardingType sharding_;
    std::unique_ptr<DTensor> weight_;
};

// =============================================================================
// Distributed Vocab-Parallel Embedding
// =============================================================================

/**
 * @class DEmbeddingVParallel
 * @brief Vocab-parallel embedding: each rank owns vocab_size/world_size rows
 * 
 * Forward: mask input → local lookup → zero out-of-shard → return partial embeddings
 * Caller must AllReduce (via sync_w_autograd) to assemble the full result.
 */
class DEmbeddingVParallel : public DModuleBase {
public:
    int64_t vocab_size_;
    int64_t embedding_dim_;
    std::unique_ptr<DTensor> weight;
    
    int64_t local_v_;
    int64_t vocab_start_;
    int64_t vocab_end_;

    DEmbeddingVParallel(const DeviceMesh& mesh,
                        std::shared_ptr<ProcessGroupNCCL> pg,
                        int64_t vocab_size,
                        int64_t embedding_dim,
                        float sd = 0.02f,
                        int seed = 42);

    OwnTensor::Tensor forward_tensor(OwnTensor::Tensor& input);

    void all_reduce_gradients(ProcessGroupNCCL* pg) override;

private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
};

// =============================================================================
// Distributed LM Head (supports weight tying)
// =============================================================================

/**
 * @class DLMHead
 * @brief Output projection head for language models
 * 
 * Two modes:
 * - Weight tying: shares embedding weight (transposed), no own parameters
 * - Separate: creates own [vocab_size, n_embd] weight matrix
 */
class DLMHead : public DModuleBase {
public:
    int64_t batch_size_;
    int64_t seq_len_;
    int64_t in_features_;
    int64_t vocab_size_;
    std::unique_ptr<DTensor> weight;
    bool use_tied_weights;
    DTensor* tied_weight_;  // Pointer to embedding.weight

    DLMHead() = default;

    // Constructor WITH weight tying
    DLMHead(const DeviceMesh& mesh,
            std::shared_ptr<ProcessGroupNCCL> pg,
            int64_t batch_size,
            int64_t seq_len,
            int64_t in_features,
            int64_t vocab_size,
            bool use_tied_weights,
            DTensor* embedding_weight);

    // Constructor WITHOUT weight tying (separate weight)
    DLMHead(const DeviceMesh& mesh,
            std::shared_ptr<ProcessGroupNCCL> pg,
            int64_t batch_size,
            int64_t seq_len,
            int64_t in_features,
            int64_t vocab_size,
            bool use_tied_weights,
            const std::vector<float>& weight_data);

    std::vector<DTensor*> parameters() override;
    DTensor forward(DTensor& input) override;

private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
};

// =============================================================================
// Distributed Layer Normalization
// =============================================================================

class DLayerNorm : public DModuleBase {
public:
    DLayerNorm(const DeviceMesh& mesh, int64_t dim, bool has_bias = true);
    
    DTensor forward(DTensor& input) override;
    void to(OwnTensor::DeviceIndex dev) override;
    void all_reduce_gradients(ProcessGroupNCCL* pg) override;

private:
    const DeviceMesh* mesh_;
    OwnTensor::nn::LayerNorm ln_;
    bool has_bias_;
};

// =============================================================================
// Vocab-Parallel Cross Entropy
// =============================================================================

OwnTensor::Tensor vocab_parallel_cross_entropy(DTensor& logits_dt, OwnTensor::Tensor& targets);

// =============================================================================
// Standard Replicated Cross Entropy Loss
// =============================================================================

class CrossEntropyLoss {
public:
    OwnTensor::Tensor forward(const OwnTensor::Tensor& logits, const OwnTensor::Tensor& targets);
};

// =============================================================================
// Optimizers
// =============================================================================

class SGD {
public:
    explicit SGD(float lr = 0.01f) : lr_(lr) {}
    void step(std::vector<DTensor*> params, ProcessGroupNCCL* pg);
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }
private:
    float lr_;
};

class AdamW {
public:
    explicit AdamW(float lr, float beta1 = 0.9f, float beta2 = 0.999f,
                   float eps = 1e-8f, float weight_decay = 0.01f)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
          weight_decay_(weight_decay), t_(0) {}
    
    void step(std::vector<DTensor*> params);
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }

    void zero_grad() {
        // AdamW doesn't own params, call zero_grad on parameters directly
    }

private:
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    int t_;
    std::unordered_map<DTensor*, OwnTensor::Tensor> m_;
    std::unordered_map<DTensor*, OwnTensor::Tensor> v_;
};

} // namespace CustomDNN



