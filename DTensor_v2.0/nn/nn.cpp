#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <unordered_set>
// #include <dlfcn.h>  
// #include "/home/blu-bridge015/Desktop/Data Parallel/Tensor-Implementations/include/TensorLib.h"
#include "TensorLib.h"
#include "nn/nn.hpp"
#include <cuda_runtime.h>
// #include "/home/blu-bridge015/Desktop/dist/include/ProcessGroupNCCL.h"
// #include "/home/blu-bridge015/Desktop/dist/Tensor-Implementations/include/TensorLib.h"





Linear::Linear(int input_dimensions, int output_dimensions, bool bias, OwnTensor::Dtype dtype)
    :input_dimensions_(input_dimensions),
    output_dimensions_(output_dimensions),
    bias_(bias){
        OwnTensor::TensorOptions opts;
        opts.dtype = dtype;
        opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CPU);

        // if(!w_.is_valid()){
        //     w_ = OwnTensor::Tensor::randn(OwnTensor::Shape{{input_dimensions_, output_dimensions_}}, opts);
        //     grad_w_ = OwnTensor::Tensor::zeros(OwnTensor::Shape{{input_dimensions_, output_dimensions_}}, opts);
        //     if(bias_){
        //         b_ = OwnTensor::Tensor::randn(OwnTensor::Shape{{1, output_dimensions_}}, opts);
        //         grad_b_ = OwnTensor::Tensor::zeros(OwnTensor::Shape{{1, output_dimensions_}}, opts);
        //     }
        // }

        w_ = new Params(OwnTensor::Tensor::randn(OwnTensor::Shape{{input_dimensions_, output_dimensions_}}, opts, 42, 1.0f));
        if(bias_){
            b_ = new Params(OwnTensor::Tensor::randn(OwnTensor::Shape{{1, output_dimensions_}}, opts, 42, 1.0f));
        }
        
        name_to_params["w1"] = w_;
        name_to_params["b1"] =b_;
        // w_.display();
    }

Params Linear::forward(OwnTensor::Tensor input_tensor, NNNode* parent, std::vector<NNNode*>& graph){

    // std::vector
    
        output_ = new Params(OwnTensor::Tensor(input_tensor.shape(), {input_tensor.dtype(), input_tensor.device()}));
    
    if(input_tensor.shape().dims[1] != input_dimensions_){
        throw std::runtime_error(
            "Mismatch in dimensions!!"
        );
    }

    if(w_->tensor_.is_cpu() && input_tensor.is_cuda()){
        throw std::runtime_error(
            "Both the tensor is not in the same gpu"
        );
    }

    output_->tensor_ =  OwnTensor::matmul(input_tensor, w_->tensor_);
    if(bias_) output_->tensor_ += b_->tensor_;

    NNNode* node = new NNNode();
    node -> op = this;
    if (parent) {
        node->parents.push_back(parent);
        node->parent_inputs.push_back(parent->output);
    }

    node->output = output_->tensor_;
    node->grad = OwnTensor::Tensor::zeros(
        node->output.shape(),
        {node->output.dtype(), node->output.device()}
    );
    graph.push_back(node);
    return *output_;
}

void Linear::to(OwnTensor::DeviceIndex device){
    
    if(w_->tensor_.is_valid()){  
        // w_.to(device);
        OwnTensor::Tensor a = w_->tensor_.to(device);
        w_->tensor_ = a;
        a = w_->tensor_grad_.to(device);
        w_ ->tensor_grad_ = a;
        if(bias_) {
            a = b_->tensor_.to(device);
            b_->tensor_ = a;

            a = b_ ->tensor_grad_.to(device);
            b_->tensor_grad_ = a;
        }

        
    }else{
        throw std::runtime_error(
            "Tensors are not valid to be moved!!"
        );
    }
    return;
}

void Linear::backward(NNNode* node){
    for (size_t i = 0; i < node->parents.size(); ++i) {
        NNNode* parent = node->parents[i];
        const OwnTensor::Tensor& x = node->parent_inputs[i];

        // dL/dX = dL/dY · Wᵀ
        parent->grad += matmul(node->grad, w_->tensor_.transpose(0, 1));

        // dL/dW = Xᵀ · dL/dY
        w_->tensor_grad_ += matmul(x.transpose(0, 1), node->grad);
    }
    if (bias_) {
        b_->tensor_grad_ += node->grad;
    }
}


MLP::MLP(std::vector<Linear> linear_vector):linear_(linear_vector) {

    if(linear_.size() == 0){
        throw std::runtime_error("No Layers within the MLP!!");
    }
    for(int i = 0; i < linear_.size(); i++){
        std::string name = std::string("l") + std::to_string(i); 
        name_to_module[name] = &linear_[i];
    }
}

void MLP::to(OwnTensor::DeviceIndex device){
    for(auto& linear: linear_){
        linear.to(device);
    }
}

OwnTensor::Tensor MLP::forward(OwnTensor::Tensor input){
    graph_.clear();
    NNNode* prev = nullptr;
    for(auto& linear: linear_){
        input = linear.forward(input, prev, graph_).tensor_;
        prev = graph_.back();
    }
    return input;
}

void MLP::backward(){
    topo_order_.clear();
    std::unordered_set<NNNode*> visited;
    topo_sort(graph_.back(), visited, topo_order_);
    std::reverse(topo_order_.begin(), topo_order_.end());
    NNNode* out = topo_order_[0]; 
    OwnTensor::TensorOptions opts;
    opts.dtype  = out->output.dtype();
    opts.device = out->output.device();
    out->grad = OwnTensor::Tensor::ones(
        out->output.shape(),
        opts
    );
    // linear->gradient_ = OwnTensor::Tensor::ones(linear->w_.shape(), opts);
    for (auto it = topo_order_.begin(); it != topo_order_.end(); ++it) {
        (*it)->op->backward(*it);
    }
        
}


// =============================================================================
// DTensor-based DLinear Implementation
// =============================================================================

#include <unparalleled/unparalleled.h>
#include <cmath>

DLinear::DLinear(int64_t in_features,
                 int64_t out_features,
                 std::shared_ptr<DeviceMesh> mesh,
                 std::shared_ptr<ProcessGroupNCCL> pg,
                 ParallelType parallel_type)
    : in_features_(in_features),
      out_features_(out_features),
      mesh_(mesh),
      pg_(pg),
      parallel_type_(parallel_type),
      weight_(nullptr)
{
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
    
    // For row-parallel, we need AllReduce to sum partial results
    if (parallel_type_ == ParallelType::ROW) {
        output.sync();  // AllReduce sum
    }
    
    return output;
}

DTensor& DLinear::weight() {
    return *weight_;
}

void DLinear::set_requires_grad(bool requires) {
    weight_->set_requires_grad(requires);
}

void DLinear::zero_grad() {
    weight_->zero_grad();
}

// =============================================================================
// DTensor-based DMLP Implementation
// =============================================================================

DMLP::DMLP(int64_t in_features,
           int64_t hidden_features,
           int64_t out_features,
           std::shared_ptr<DeviceMesh> mesh,
           std::shared_ptr<ProcessGroupNCCL> pg)
{
    fc1_ = std::make_unique<DLinear>(in_features, hidden_features, mesh, pg, ParallelType::COLUMN);
    fc2_ = std::make_unique<DLinear>(hidden_features, out_features, mesh, pg, ParallelType::ROW);
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

DLinear& DMLP::fc1() { return *fc1_; }
DLinear& DMLP::fc2() { return *fc2_; }

// =============================================================================
// DLinearReplicated Implementation
// =============================================================================

DLinearReplicated::DLinearReplicated(int64_t in_features,
                                     int64_t out_features,
                                     std::shared_ptr<DeviceMesh> mesh,
                                     std::shared_ptr<ProcessGroupNCCL> pg)
    : in_features_(in_features), out_features_(out_features), mesh_(mesh), pg_(pg) {
    
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

// =============================================================================
// DEmbedding Implementation
// =============================================================================

DEmbedding::DEmbedding(int64_t vocab_size,
                       int64_t embedding_dim,
                       std::shared_ptr<DeviceMesh> mesh,
                       std::shared_ptr<ProcessGroupNCCL> pg)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim),
      mesh_(mesh), pg_(pg)
{
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

void DEmbedding::set_requires_grad(bool requires) {
    weight_->set_requires_grad(requires);
}

void DEmbedding::zero_grad() {
    weight_->zero_grad();
}
