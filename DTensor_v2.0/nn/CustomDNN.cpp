#include "nn/CustomDNN.h"
#include <unparalleled/unparalleled.h>
#include "ops/helpers/AdamKernels.h"
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

    // Xavier/Kaiming-like initialization: std = 1/sqrt(in_features)
    float std_scale = 1.0f / std::sqrt(static_cast<float>(in_features_));
    auto w_data = weight_->getData();
    for (size_t i = 0; i < w_data.size(); ++i) {
        w_data[i] *= std_scale;
    }
    weight_->setData(w_data, weight_->layout());
    weight_->set_requires_grad(true);
    
    // Initialize bias if needed
    // For column-parallel (Shard(1)), output is sharded on last dim, so bias must match
    // For row-parallel (Shard(0)), output is replicated after sync, so bias should be replicated
    if (has_bias_) {
        // Linear layer output: Y = X @ W
        // Col Parallel: X [B, K] @ W [K, N/P] -> Y [B, N/P] (Sharded on dim 1)
        // Row Parallel: X [B, K/P] @ W [K/P, N] -> Y [B, N] (Replicated after sync)
        
        // Bias shape is [N] (out_features)
        
        if (weight_sharding_.is_shard() && weight_sharding_.shard_dim() == 1) {
            // Column-parallel: Output is [B, N/P]. Bias should be [1, N/P] to broadcast correctly.
            // We create a 2D bias tensor [1, N] sharded on dim 1: [1] -> [1], [N] -> [N/P] local.
            Layout b_layout(*mesh, {1, out_features_}, 1);  // 2D, sharded on dim 1
            bias_ = std::make_unique<DTensor>(
                DTensor::zeros({1, out_features_}, mesh, pg, b_layout));
        } else {
            // Row-parallel: Output is [B, N] (Replicated). Bias should be [1, N] (Replicated).
            Layout b_layout = Layout::replicated(*mesh, {1, out_features_});  // 2D  
            bias_ = std::make_unique<DTensor>(
                DTensor::zeros({1, out_features_}, mesh, pg, b_layout));
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
    
    // Xavier initialization: scale weights by 1/sqrt(in_features)
    float scale = 1.0f / std::sqrt(static_cast<float>(in_features_));
    auto w_data = weight_->getData();
    for (auto& v : w_data) v *= scale;
    weight_->setData(w_data, weight_->get_layout());
    
    weight_->set_requires_grad(true);
}

DTensor DLinear::forward(const DTensor& input) {
    // Compute Y = X @ W
    DTensor output = input.matmul(*weight_);
    
    // Auto-sync: Row-parallel (Shard(0)) needs AllReduce to sum partial results
    if (weight_sharding_.is_shard() && weight_sharding_.shard_dim() == 0) {
        output.sync();  // AllReduce sum
    }

    // Add bias if present
    // Note: Add bias AFTER sync for row-parallel to avoid adding bias multiple times
    if (has_bias_ && bias_) {
        output = output.add(*bias_);
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
        mesh, pg, in_features, hidden_features,
        ShardingType::Shard(1),      // Weight: Column-sharded
        ShardingType::Replicated(),  // Bias: Replicated (adapted internally)
        false); // Use false to disable bias
    
    // Second layer: Row-parallel (shards input to produce replicated output)
    fc2_ = std::make_unique<DLinear>(
        mesh, pg, hidden_features, out_features,
        ShardingType::Shard(0),      // Weight: Row-sharded
        ShardingType::Replicated(),  // Bias: Replicated
        false); // Use false to disable bias
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
    // std::cout << "[DEBUG] DEmbedding: Creating layout..." << std::endl;
    Layout weight_layout = Layout::replicated(*mesh, {vocab_size, embedding_dim});
    
    // std::cout << "[DEBUG] DEmbedding: Creating weight DTensor..." << std::endl;
    weight_ = std::make_unique<DTensor>(
        DTensor::randn({vocab_size, embedding_dim}, mesh, pg, weight_layout)
    );
    
    // std::cout << "[DEBUG] DEmbedding: Scaling weight..." << std::endl;
    // Scale initialization (Xavier-like)
    float scale = 1.0f / std::sqrt(static_cast<float>(embedding_dim));
    auto weight_data = weight_->getData();
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] *= scale;
    }
    // std::cout << "[DEBUG] DEmbedding: Setting weight data..." << std::endl;
    weight_->setData(weight_data, weight_layout);
    
    // std::cout << "[DEBUG] DEmbedding: Setting requires_grad..." << std::endl;
    weight_->set_requires_grad(true);
    // std::cout << "[DEBUG] DEmbedding: Constructor done." << std::endl;
}

DTensor DEmbedding::forward(const std::vector<int>& token_ids) {
    int batch_size = token_ids.size();
    
    // Create indices tensor on the same device as weight
    OwnTensor::Tensor indices_tensor(OwnTensor::Shape{{static_cast<int64_t>(batch_size)}}, 
                                     OwnTensor::TensorOptions()
                                         .with_device(weight_->local_tensor().device())
                                         .with_dtype(OwnTensor::Dtype::Int32));
    indices_tensor.set_data(token_ids.data(), token_ids.size());
    
    // Call efficient embedding lookup
    return DTensor::embedding(indices_tensor, *weight_);
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

void AdamW::step(std::vector<DTensor*> params) {
    if (params.empty()) return;
    auto pg = params[0]->get_pg();
    t_++;

    float bias_corr1 = 1.0f - std::pow(beta1_, t_);
    float bias_corr2 = 1.0f - std::pow(beta2_, t_);

    // Gradient clipping for stability
    float global_norm_sq = 0.0f;
    const float max_norm = 1.0f;
    
    for (DTensor* param : params) {
        if (!param->requires_grad()) continue;
        OwnTensor::Tensor grad = param->grad();
        float local_norm_sq = 0.0f;
        auto grad_cpu = grad.to_cpu();
        const float* g_ptr = grad_cpu.data<float>();
        for (size_t i = 0; i < grad.numel(); ++i) {
            local_norm_sq += g_ptr[i] * g_ptr[i];
        }
        global_norm_sq += local_norm_sq;
    }
    
    float global_sum_sq = 0.0f;
    pg->all_reduce_cpu(&global_norm_sq, &global_sum_sq, 1, OwnTensor::Dtype::Float32, op_t::sum);
    
    float global_norm = std::sqrt(global_sum_sq);
    float clip_coef = (global_norm > max_norm) ? (max_norm / global_norm) : 1.0f;

    for (DTensor* param : params) {
        if (!param->requires_grad()) continue;

        OwnTensor::Tensor& weight = param->local_tensor();
        OwnTensor::Tensor grad = param->grad();

        // 1. Synchronize gradients for Replicated parameters
        if (param->get_layout().is_replicated()) {
             pg->all_reduce(grad.data<float>(), grad.data<float>(), 
                            grad.numel(), OwnTensor::Dtype::Float32, op_t::sum, true);
        }

        // 2. Initialize state if needed
        if (m_.find(param) == m_.end()) {
            m_[param] = OwnTensor::Tensor::zeros(weight.shape(), weight.opts());
            v_[param] = OwnTensor::Tensor::zeros(weight.shape(), weight.opts());
        }

        OwnTensor::Tensor& m = m_[param];
        OwnTensor::Tensor& v = v_[param];

        // 3. Launch fused AdamW kernel
        // We use clip_coef * lr_ as the effective learning rate to incorporate clipping
        OwnTensor::cuda::fused_adam_cuda(
            weight.data<float>(),
            grad.data<float>(),
            m.data<float>(),
            v.data<float>(),
            weight.numel(),
            lr_ * clip_coef,
            beta1_,
            beta2_,
            eps_,
            weight_decay_,
            bias_corr1,
            bias_corr2
        );
    }
}

// =============================================================================
// CrossEntropyLoss Implementation
// =============================================================================

CrossEntropyLoss::CrossEntropyLoss(std::shared_ptr<DeviceMesh> mesh,
                                   std::shared_ptr<ProcessGroupNCCL> pg)
{
    mesh_ = mesh;
    pg_ = pg;
}

DTensor CrossEntropyLoss::forward(const DTensor& logits, const DTensor& targets) {
    if (logits.get_layout().is_sharded()) {
        return logits.distributed_sparse_cross_entropy_loss(targets);
    }
    return logits.sparse_cross_entropy_loss(targets);
}

// =============================================================================
// DLayerNorm Implementation
// =============================================================================

DLayerNorm::DLayerNorm(int normalized_shape,
                       std::shared_ptr<DeviceMesh> mesh,
                       std::shared_ptr<ProcessGroupNCCL> pg,
                       float eps)
    : normalized_shape_(normalized_shape), eps_(eps)
{
    mesh_ = mesh;
    pg_ = pg;

    Layout repl_layout = Layout::replicated(*mesh_, {(int64_t)normalized_shape_});
    
    // Initialize weight to ones and bias to zeros
    weight_ = std::make_unique<DTensor>(DTensor::ones({(int64_t)normalized_shape_}, mesh_, pg_, repl_layout));
    bias_ = std::make_unique<DTensor>(DTensor::zeros({(int64_t)normalized_shape_}, mesh_, pg_, repl_layout));
}

DTensor DLayerNorm::forward(const DTensor& input) {
    return input.layer_norm(*weight_, *bias_, eps_);
}

void DLayerNorm::set_requires_grad(bool requires) {
    weight_->set_requires_grad(requires);
    bias_->set_requires_grad(requires);
}

void DLayerNorm::zero_grad() {
    weight_->zero_grad();
    bias_->zero_grad();
}

std::vector<DTensor*> DLayerNorm::parameters() {
    return {weight_.get(), bias_.get()};
}

} // namespace dnn
} // namespace OwnTensor
