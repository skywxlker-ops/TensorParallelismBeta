#include "nn/CustomDNN.h"
#include <unparalleled/unparalleled.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

namespace OwnTensor {
namespace dnn {

// =============================================================================
// DLinear Implementation
// =============================================================================

// NEW API: Constructor with explicit ShardingType for weight and bias
DLinear::DLinear(std::shared_ptr<DeviceMesh> mesh,
                 std::shared_ptr<ProcessGroupNCCL> pg,
                 int64_t in_features,
                 int64_t out_features,
                 ShardingType weight_sharding,
                 ShardingType bias_sharding,
                 bool has_bias)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(has_bias),
      weight_sharding_(weight_sharding),
      bias_sharding_(bias_sharding),
      parallel_type_(weight_sharding.is_shard() && weight_sharding.shard_dim() == 0 
                     ? ParallelType::ROW : ParallelType::COLUMN),
      weight_(nullptr),
      bias_(nullptr)
{
    mesh_ = mesh;
    pg_ = pg;
    
    // Initialize weight based on sharding type
    if (weight_sharding_.is_shard()) {
        Layout w_layout(*mesh, {in_features_, out_features_}, weight_sharding_.shard_dim());
        weight_ = std::make_unique<DTensor>(
            DTensor::randn({in_features_, out_features_}, mesh, pg, w_layout));
    } else {
        // Replicated weight
        Layout w_layout = Layout::replicated(*mesh, {in_features_, out_features_});
        weight_ = std::make_unique<DTensor>(
            DTensor::randn({in_features_, out_features_}, mesh, pg, w_layout));
    }
    weight_->set_requires_grad(true);
    
    // Initialize bias if needed
    if (has_bias_) {
        if (bias_sharding_.is_shard()) {
            Layout b_layout(*mesh, {out_features_}, bias_sharding_.shard_dim());
            bias_ = std::make_unique<DTensor>(
                DTensor::zeros({out_features_}, mesh, pg, b_layout));
        } else {
            // Replicated bias
            Layout b_layout = Layout::replicated(*mesh, {out_features_});
            bias_ = std::make_unique<DTensor>(
                DTensor::zeros({out_features_}, mesh, pg, b_layout));
        }
        bias_->set_requires_grad(true);
    }
}

// LEGACY: Constructor with ParallelType (backward compatible)
DLinear::DLinear(int64_t in_features,
                 int64_t out_features,
                 std::shared_ptr<DeviceMesh> mesh,
                 std::shared_ptr<ProcessGroupNCCL> pg,
                 ParallelType parallel_type)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(false),
      weight_sharding_(parallel_type == ParallelType::COLUMN 
                       ? ShardingType::Shard(1) : ShardingType::Shard(0)),
      bias_sharding_(ShardingType::Replicated()),
      parallel_type_(parallel_type),
      weight_(nullptr),
      bias_(nullptr)
{
    mesh_ = mesh;
    pg_ = pg;
    
    // Initialize weight with appropriate layout for parallelism type
    if (parallel_type_ == ParallelType::COLUMN) {
        // Column parallel: shard output dimension (dim 1)
        Layout w_layout(*mesh, {in_features_, out_features_}, 1);
        weight_ = std::make_unique<DTensor>(
            DTensor::randn({in_features_, out_features_}, mesh, pg, w_layout));
    } else {
        // Row parallel: shard input dimension (dim 0)
        Layout w_layout(*mesh, {in_features_, out_features_}, 0);
        weight_ = std::make_unique<DTensor>(
            DTensor::randn({in_features_, out_features_}, mesh, pg, w_layout));
    }
    weight_->set_requires_grad(true);
}

DTensor DLinear::forward(const DTensor& input) {
    // Compute Y = X @ W
    DTensor output = input.matmul(*weight_);
    
    // Add bias if present
    if (has_bias_ && bias_) {
        output = output.add(*bias_);
    }
    
    // Auto-sync: Row-parallel (Shard(0)) needs AllReduce to sum partial results
    if (weight_sharding_.is_shard() && weight_sharding_.shard_dim() == 0) {
        output.sync();  // AllReduce sum
    }
    
    return output;
}

DTensor& DLinear::weight() {
    return *weight_;
}

DTensor& DLinear::bias() {
    if (!has_bias_ || !bias_) {
        throw std::runtime_error("DLinear: bias() called but has_bias=false");
    }
    return *bias_;
}

void DLinear::set_requires_grad(bool requires) {
    weight_->set_requires_grad(requires);
    if (has_bias_ && bias_) {
        bias_->set_requires_grad(requires);
    }
}

void DLinear::zero_grad() {
    weight_->zero_grad();
    if (has_bias_ && bias_) {
        bias_->zero_grad();
    }
}

std::vector<DTensor*> DLinear::parameters() {
    std::vector<DTensor*> params = {weight_.get()};
    if (has_bias_ && bias_) {
        params.push_back(bias_.get());
    }
    return params;
}

// =============================================================================
// DLinearReplicated Implementation
// =============================================================================

DLinearReplicated::DLinearReplicated(int64_t in_features,
                                     int64_t out_features,
                                     std::shared_ptr<DeviceMesh> mesh,
                                     std::shared_ptr<ProcessGroupNCCL> pg)
    : in_features_(in_features),
      out_features_(out_features),
      weight_(nullptr)
{
    mesh_ = mesh;
    pg_ = pg;
    
    // Replicated weight: [in_features, out_features] - full matrix on each GPU
    Layout w_layout = Layout::replicated(*mesh, {in_features, out_features});
    
    // Xavier/He initialization: scale = sqrt(2/in_features)
    weight_ = std::make_unique<DTensor>(
        DTensor::randn({in_features_, out_features_}, mesh, pg, w_layout));
    
    // Scale weights for better initialization
    float scale = std::sqrt(2.0f / in_features_);
    auto w_data = weight_->getData();
    for (auto& v : w_data) v *= scale;
    weight_->setData(w_data, w_layout);
    
    weight_->set_requires_grad(true);
}

DTensor DLinearReplicated::forward(const DTensor& input) {
    // Simple matmul: Y = X @ W (both replicated, autograd-aware)
    return input.matmul(*weight_);
}

DTensor& DLinearReplicated::weight() {
    return *weight_;
}

void DLinearReplicated::set_requires_grad(bool requires) {
    weight_->set_requires_grad(requires);
}

void DLinearReplicated::zero_grad() {
    weight_->zero_grad();
}

std::vector<DTensor*> DLinearReplicated::parameters() {
    return {weight_.get()};
}

// =============================================================================
// DMLP Implementation
// =============================================================================

DMLP::DMLP(int64_t in_features,
           int64_t hidden_features,
           int64_t out_features,
           std::shared_ptr<DeviceMesh> mesh,
           std::shared_ptr<ProcessGroupNCCL> pg)
{
    mesh_ = mesh;
    pg_ = pg;
    
    // First layer: Column-parallel (shards hidden dimension)
    fc1_ = std::make_unique<DLinear>(
        in_features, hidden_features, mesh, pg, ParallelType::COLUMN);
    
    // Second layer: Row-parallel (shards input to produce replicated output)
    fc2_ = std::make_unique<DLinear>(
        hidden_features, out_features, mesh, pg, ParallelType::ROW);
}

DTensor DMLP::forward(const DTensor& input) {
    // Layer 1: Column-parallel matmul
    DTensor h = fc1_->forward(input);
    
    // Activation: GeLU (better for transformers)
    DTensor h_act = h.gelu();
    
    // Layer 2: Row-parallel matmul (includes AllReduce)
    DTensor output = fc2_->forward(h_act);
    
    return output;
}

void DMLP::set_requires_grad(bool requires) {
    fc1_->set_requires_grad(requires);
    fc2_->set_requires_grad(requires);
}

void DMLP::zero_grad() {
    fc1_->zero_grad();
    fc2_->zero_grad();
}

std::vector<DTensor*> DMLP::parameters() {
    auto params1 = fc1_->parameters();
    auto params2 = fc2_->parameters();
    params1.insert(params1.end(), params2.begin(), params2.end());
    return params1;
}

DLinear& DMLP::fc1() { 
    return *fc1_; 
}

DLinear& DMLP::fc2() { 
    return *fc2_; 
}

// =============================================================================
// DEmbedding Implementation
// =============================================================================

DEmbedding::DEmbedding(int64_t vocab_size,
                       int64_t embedding_dim,
                       std::shared_ptr<DeviceMesh> mesh,
                       std::shared_ptr<ProcessGroupNCCL> pg)
    : vocab_size_(vocab_size),
      embedding_dim_(embedding_dim),
      weight_(nullptr)
{
    mesh_ = mesh;
    pg_ = pg;
    
    // Create replicated embedding weight [vocab_size, embedding_dim]
    // We replicate for simplicity (sharding would require all-gather during lookup)
    Layout weight_layout = Layout::replicated(*mesh, {vocab_size, embedding_dim});
    
    weight_ = std::make_unique<DTensor>(
        DTensor::randn({vocab_size, embedding_dim}, mesh, pg, weight_layout)
    );
    
    // Scale initialization (Xavier-like)
    float scale = 1.0f / std::sqrt(static_cast<float>(embedding_dim));
    auto weight_data = weight_->getData();
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] *= scale;
    }
    weight_->setData(weight_data, weight_layout);
    
    weight_->set_requires_grad(true);
}

DTensor DEmbedding::forward(const std::vector<int>& token_ids) {
    int batch_size = token_ids.size();
    
    // Create indices tensor on GPU (uint16 dtype as required by OwnTensor embedding)
    std::vector<uint16_t> indices_u16(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        int token = token_ids[i];
        if (token < 0 || token >= vocab_size_) token = 0;
        indices_u16[i] = static_cast<uint16_t>(token);
    }
    
    // Create OwnTensor for indices on GPU
    OwnTensor::Shape idx_shape;
    idx_shape.dims = {static_cast<int64_t>(batch_size)};
    OwnTensor::TensorOptions opts = OwnTensor::TensorOptions()
        .with_device(weight_->local_tensor().device())
        .with_dtype(OwnTensor::Dtype::UInt16);
    OwnTensor::Tensor indices = OwnTensor::Tensor::zeros(idx_shape, opts);
    
    // Copy indices to GPU
    cudaMemcpy(indices.data<uint16_t>(), indices_u16.data(), 
               batch_size * sizeof(uint16_t), cudaMemcpyHostToDevice);
    
    // Use DTensor::embedding for autograd-aware lookup
    return DTensor::embedding(indices, *weight_, -1);
}

DTensor& DEmbedding::weight() {
    return *weight_;
}

void DEmbedding::set_requires_grad(bool requires) {
    weight_->set_requires_grad(requires);
}

void DEmbedding::zero_grad() {
    weight_->zero_grad();
}

std::vector<DTensor*> DEmbedding::parameters() {
    return {weight_.get()};
}

// =============================================================================
// SGD Optimizer Implementation
// =============================================================================

void SGD::step(std::vector<DTensor*> params) {
    // Gradient clipping: compute global norm and scale if needed
    float global_norm_sq = 0.0f;
    const float max_norm = 1000.0f;  // Balanced clip threshold
    
    // First pass: compute global gradient norm
    for (DTensor* param : params) {
        if (!param->requires_grad()) continue;
        auto grad_cpu = param->grad().to_cpu();
        const float* g_ptr = grad_cpu.data<float>();
        size_t numel = grad_cpu.numel();
        for (size_t i = 0; i < numel; ++i) {
            global_norm_sq += g_ptr[i] * g_ptr[i];
        }
    }
    float global_norm = std::sqrt(global_norm_sq);
    float clip_coef = (global_norm > max_norm) ? (max_norm / global_norm) : 1.0f;
    
    // Second pass: apply clipped gradient update
    for (DTensor* param : params) {
        if (!param->requires_grad()) continue;
        
        // Get gradient and weight tensors
        OwnTensor::Tensor grad = param->grad();
        
        // Copy weight to CPU for update (simple implementation)
        OwnTensor::Tensor weight_cpu = param->local_tensor().to_cpu();
        auto grad_cpu = grad.to_cpu();
        
        float* w_ptr = weight_cpu.data<float>();
        const float* g_ptr = grad_cpu.data<float>();
        size_t numel = weight_cpu.numel();
        
        // W = W - lr * clip_coef * grad
        for (size_t i = 0; i < numel; ++i) {
            w_ptr[i] -= lr_ * clip_coef * g_ptr[i];
        }
        
        // Copy back to GPU using cudaMemcpy
        cudaMemcpy(param->local_tensor().data<float>(), w_ptr, 
                   numel * sizeof(float), cudaMemcpyHostToDevice);
    }
}

} // namespace dnn
} // namespace OwnTensor
