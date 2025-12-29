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

        w_ = new Params(OwnTensor::Tensor::randn(OwnTensor::Shape{{input_dimensions_, output_dimensions_}}, opts));
        if(bias_){
            b_ = new Params(OwnTensor::Tensor::randn(OwnTensor::Shape{{1, output_dimensions_}}, opts));
        }
        
        name_to_params["w1"] = w_;
        name_to_params["b1"] =b_;
        // w_.display();
    }

Params Linear::forward(OwnTensor::Tensor input_tensor, Node* parent, std::vector<Node*>& graph){

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

    Node* node = new Node();
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

void Linear::backward(Node* node){
    for (size_t i = 0; i < node->parents.size(); ++i) {
        Node* parent = node->parents[i];
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
    Node* prev = nullptr;
    for(auto& linear: linear_){
        input = linear.forward(input, prev, graph_).tensor_;
        prev = graph_.back();
    }
    return input;
}

void MLP::backward(){
    topo_order_.clear();
    std::unordered_set<Node*> visited;
    topo_sort(graph_.back(), visited, topo_order_);
    std::reverse(topo_order_.begin(), topo_order_.end());
    Node* out = topo_order_[0]; 
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


int main(){

    MLP mlp({
        Linear(20,30, true),
        Linear(30,20)
    });
    mlp.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, 0));
    cudaDeviceSynchronize();
    OwnTensor::TensorOptions opts;
    opts.dtype = OwnTensor::Dtype::Float32;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, 0);

    OwnTensor::Tensor input = OwnTensor::Tensor::randn({{20,20}},opts);
    input.to_cpu().display();
    input.to(opts.device);

    input = mlp.forward(input);
    input.to_cpu().display();
    std::unordered_map<std::string, Params*> params = mlp.collect_named_params();
    // for(auto& [name, parameters] : params){
    //     std::cout<<name <<":\n ";    
    //     // parameters->tensor_.to_cpu().display();
    //     // parameters->tensor_.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, 0));
    // }
    std::cout << "l0.w1: \n";
    params["l0.w1"]->tensor_.to_cpu().display();
    std::cout << "l1.w1: \n";
    params["l1.w1"]->tensor_.to_cpu().display();
    std::cout << "l0.b1: \n";
    params["l0.b1"]->tensor_.to_cpu().display();


    mlp.backward();

    // input.to_cpu().display();
    return 0;
}
