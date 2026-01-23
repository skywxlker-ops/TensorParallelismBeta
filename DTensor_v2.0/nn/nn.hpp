#pragma once
/**
 * @file nn.hpp
 * @brief Neural Network Module
 * 
 * Supports both:
 * - OwnTensor::Tensor for single-GPU training
 * - DTensor for distributed tensor-parallel training
 */

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "TensorLib.h"

// Forward declare DTensor (will be included in .cpp)
class DTensor;
class DeviceMesh;
class ProcessGroupNCCL;

// =============================================================================
// Tensor Metadata
// =============================================================================

struct _TensorMeta {
    OwnTensor::Shape  _tensor_shape;
    OwnTensor::Dtype  _tensor_dtype;
    OwnTensor::Stride _tensor_stride;
    OwnTensor::DeviceIndex _tensor_device;
    uint64_t _ndim;
    uint64_t _nbytes;
};

// =============================================================================
// Parameter Container
// =============================================================================

struct Params {
    OwnTensor::Tensor tensor_;
    OwnTensor::Tensor tensor_grad_;
    Params() = default;
    _TensorMeta* _tensor_meta = new _TensorMeta();

    Params(OwnTensor::Tensor tensor)
        : tensor_(tensor),
          tensor_grad_(OwnTensor::Tensor::zeros(tensor_.shape(), {tensor_.dtype(), tensor_.device()}))
    {
        _tensor_meta->_tensor_shape = tensor_.shape();
        _tensor_meta->_tensor_dtype = tensor_.dtype();
        _tensor_meta->_tensor_stride = tensor_.stride();
        _tensor_meta->_tensor_device = tensor_.device();
        _tensor_meta->_ndim = tensor_.ndim();
        _tensor_meta->_nbytes = [this]()->int64_t {
            return (int64_t)(_tensor_meta->_tensor_shape.dims[0] * 
                    _tensor_meta->_tensor_shape.dims[1] *
                    tensor_.dtype_size(_tensor_meta->_tensor_dtype));
        }();
    }
};

// =============================================================================
// Base Module
// =============================================================================

class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    
    std::vector<Params*> parameters() {
        std::vector<Params*> parameters_;
        collect_params(parameters_);
        return parameters_;
    }
    
    std::unordered_map<std::string, Params*> collect_named_params() {
        std::unordered_map<std::string, Params*> named_params;
        for (auto& [name, module] : name_to_module) {
            collect_params_named(named_params, name);
        }
        return named_params;
    }

protected:
    std::unordered_map<std::string, Params*> name_to_params;
    std::unordered_map<std::string, Module*> name_to_module;

private:
    void collect_params(std::vector<Params*>& total_params) {
        for (auto& [_, params] : name_to_params) {
            total_params.push_back(params);
        }
        for (auto& [_, module] : name_to_module) {
            module->collect_params(total_params);
        }
    }
    
    void collect_params_named(std::unordered_map<std::string, Params*>& named_params, 
                              std::string module_name) {
        for (auto& [layer_name, params] : name_to_params) {
            named_params[module_name + "." + layer_name] = params;
        }
        for (auto& [mod_name, module] : name_to_module) {
            module->collect_params_named(named_params, mod_name);
        }
    }
};

// =============================================================================
// Computation Graph Node (for manual autograd)
// =============================================================================

class Linear;

struct NNNode {
    Linear* op;
    std::vector<NNNode*> parents;
    std::vector<OwnTensor::Tensor> parent_inputs;
    OwnTensor::Tensor output;
    std::string name_of_tensor;
    OwnTensor::Tensor grad;
};

// =============================================================================
// Linear Layer (OwnTensor version)
// =============================================================================

class Linear : public Module {
public:
    Linear(int input_dimensions, int output_dimensions, 
           bool bias = false, OwnTensor::Dtype dtype = OwnTensor::Dtype::Float32);
    
    Params forward(OwnTensor::Tensor input_tensor, NNNode* parent, std::vector<NNNode*>& graph);
    void backward(NNNode* node);
    void to(OwnTensor::DeviceIndex device);

    OwnTensor::Tensor getWeight() { return w_->tensor_; }
    OwnTensor::Tensor getBias() { return b_->tensor_; }

private:
    int input_dimensions_;
    int output_dimensions_;
    bool requires_grad = false;
    bool bias_ = false;
    Params* w_;
    Params* b_;
    Params* output_;
    OwnTensor::TensorOptions opts_;
};

// =============================================================================
// MLP (OwnTensor version)
// =============================================================================

class MLP : public Module {
public:
    MLP(std::vector<Linear> linear_vector);
    OwnTensor::Tensor forward(OwnTensor::Tensor input);
    void backward();
    void to(OwnTensor::DeviceIndex device);

private:
    void topo_sort(NNNode* node, std::unordered_set<NNNode*>& visited, std::vector<NNNode*>& topo) {
        if (visited.count(node)) return;
        visited.insert(node);
        for (NNNode* parent : node->parents) {
            topo_sort(parent, visited, topo);
        }
        topo.push_back(node);
    }

    std::vector<NNNode*> graph_;
    std::vector<Linear> linear_;
    std::vector<NNNode*> topo_order_;
    std::vector<std::vector<OwnTensor::Tensor>> parameters_;
};

// =============================================================================
// Parallelism Type for DTensor-based layers
// =============================================================================

enum class ParallelType {
    COLUMN,  // Shard output dimension
    ROW      // Shard input dimension
};

// =============================================================================
// DTensor-based Linear Layer for Tensor Parallelism
// =============================================================================

class DLinear {
public:
    /**
     * Construct a distributed linear layer
     * @param in_features Input dimension
     * @param out_features Output dimension
     * @param mesh Device mesh
     * @param pg Process group
     * @param parallel_type Column or Row parallelism
     */
    DLinear(int64_t in_features,
            int64_t out_features,
            std::shared_ptr<DeviceMesh> mesh,
            std::shared_ptr<ProcessGroupNCCL> pg,
            ParallelType parallel_type);
    
    DTensor forward(const DTensor& input);
    
    DTensor& weight();
    void set_requires_grad(bool requires);
    void zero_grad();

private:
    int64_t in_features_;
    int64_t out_features_;
    std::shared_ptr<DeviceMesh> mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    ParallelType parallel_type_;
    std::unique_ptr<DTensor> weight_;
};

// =============================================================================
// DTensor-based Replicated Linear Layer (no parallelism, for output projection)
// =============================================================================

class DLinearReplicated {
public:
    /**
     * Construct a replicated (non-parallel) linear layer
     * Full weight matrix on each GPU for proper autograd
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
    void set_requires_grad(bool requires);
    void zero_grad();

private:
    int64_t in_features_;
    int64_t out_features_;
    std::shared_ptr<DeviceMesh> mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    std::unique_ptr<DTensor> weight_;  // [in_features, out_features] replicated
};

// =============================================================================
// DTensor-based MLP for Tensor Parallelism
// =============================================================================

class DMLP {
public:
    /**
     * Construct a 2-layer MLP with tensor parallelism
     * fc1: Column-parallel, fc2: Row-parallel
     */
    DMLP(int64_t in_features,
         int64_t hidden_features,
         int64_t out_features,
         std::shared_ptr<DeviceMesh> mesh,
         std::shared_ptr<ProcessGroupNCCL> pg);
    
    DTensor forward(const DTensor& input);
    
    void set_requires_grad(bool requires);
    void zero_grad();
    
    DLinear& fc1();
    DLinear& fc2();

private:
    std::unique_ptr<DLinear> fc1_;
    std::unique_ptr<DLinear> fc2_;
};

// =============================================================================
// DTensor-based Embedding Layer
// =============================================================================

class DEmbedding {
public:
    /**
     * Construct a distributed embedding layer
     * @param vocab_size Vocabulary size
     * @param embedding_dim Embedding dimension
     * @param mesh Device mesh
     * @param pg Process group
     */
    DEmbedding(int64_t vocab_size,
               int64_t embedding_dim,
               std::shared_ptr<DeviceMesh> mesh,
               std::shared_ptr<ProcessGroupNCCL> pg);
    
    /**
     * Forward pass: lookup embeddings for token IDs
     * @param token_ids Vector of token IDs [batch_size * seq_len]
     * @return DTensor of embeddings [batch_size * seq_len, embedding_dim]
     */
    DTensor forward(const std::vector<int>& token_ids);
    
    DTensor& weight() { return *weight_; }
    void set_requires_grad(bool requires);
    void zero_grad();

private:
    int64_t vocab_size_;
    int64_t embedding_dim_;
    std::unique_ptr<DTensor> weight_;  // [vocab_size, embedding_dim]
    std::shared_ptr<DeviceMesh> mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
};

// =============================================================================
// Simple SGD Optimizer
// =============================================================================

class SGD {
public:
    SGD(float lr) : lr_(lr) {}
    
    void step(std::vector<DTensor*> params);
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }

private:
    float lr_;
};
