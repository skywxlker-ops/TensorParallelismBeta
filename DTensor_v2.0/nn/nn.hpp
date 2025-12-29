#pragma once
#include <iostream>
#include <memory>
#include "TensorLib.h"
#include <unordered_set>
#include <unordered_map>
struct _TensorMeta{
    OwnTensor::Shape  _tensor_shape;
    OwnTensor::Dtype  _tensor_dtype;
    OwnTensor::Stride _tensor_stride;
    OwnTensor::DeviceIndex _tensor_device;
    uint64_t _ndim;
    uint64_t _nbytes;

};
struct Params{

    OwnTensor::Tensor tensor_;
    OwnTensor::Tensor tensor_grad_;
    Params() = default;
    _TensorMeta* _tensor_meta = new _TensorMeta();

    Params(OwnTensor::Tensor tensor)
    :tensor_(tensor),
    tensor_grad_(OwnTensor::Tensor::zeros(tensor_.shape(), {tensor_.dtype(), tensor_.device()}))
    {
        _tensor_meta->_tensor_shape = tensor_.shape();
        _tensor_meta->_tensor_dtype = tensor_.dtype();
        _tensor_meta->_tensor_stride = tensor_.stride();
        _tensor_meta->_tensor_device = tensor_.device();
        _tensor_meta->_ndim = tensor_.ndim();
        _tensor_meta->_nbytes = [this]()->int64_t{
            return (int64_t)(this->_tensor_meta->_tensor_shape.dims[0] * 
                    this->_tensor_meta->_tensor_shape.dims[1] *
                    this->tensor_.dtype_size(_tensor_meta->_tensor_dtype)
            );  
        }();
    }

};

class Module{
public:
    Module() = default;
    std::vector<Params*> parameters(){
        std::vector<Params*> parameters_;
        collect_params(parameters_);
        return parameters_;
    }
    std::unordered_map<std::string, Params*> collect_named_params(){
        std::unordered_map<std::string, Params*> named_params;
        for(auto& [name , module]: name_to_module){
            collect_params_named(named_params, name);
        }
        
        return named_params;
    }
protected:
    std::unordered_map<std::string, Params*> name_to_params;
    std::unordered_map<std::string, Module*> name_to_module; 

private:
    void collect_params(std::vector<Params*>& total_params){

        for(auto& [_, params]: name_to_params){
            total_params.push_back(params);
        }

        for(auto& [_, module]: name_to_module){
            module->collect_params(total_params);
        }
    }
    void collect_params_named(std::unordered_map<std::string, Params*>& named_params, std::string module_name){
        for(auto& [layer_name, params]: name_to_params){
            named_params[module_name+"."+layer_name] = params;
        }

        for(auto& [module_name, module]: name_to_module){
            module->collect_params_named(named_params, module_name);
        }
    }
};

class Linear;
struct Node {
    Linear* op;                    // which operation produced this
    std::vector<Node*> parents;    // previous nodes in graph
    std::vector<OwnTensor::Tensor> parent_inputs;

    OwnTensor::Tensor output;
    std::string name_of_tensor;
    OwnTensor::Tensor grad;        // dL/d(output)
};


class Linear: public Module{
public:
    Linear(int input_dimensions, int output_dimensions, bool bias = false, OwnTensor::Dtype dtype = OwnTensor::Dtype::Float32);
    Params forward(OwnTensor::Tensor input_tensor, Node* parent, std::vector<Node*>& graph);
    void backward(Node* node);
    void to(OwnTensor::DeviceIndex device);

    OwnTensor::Tensor getWeight(){ return w_->tensor_; }
    OwnTensor::Tensor getBias(){ return b_->tensor_; }

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

class MLP: public Module{
public:
    MLP(std::vector<Linear> linear_vector);
    OwnTensor::Tensor forward(OwnTensor::Tensor input);
    void backward();
    void to(OwnTensor::DeviceIndex device);


private:
    void topo_sort(Node* node,
               std::unordered_set<Node*>& visited,
               std::vector<Node*>& topo) {
        if (visited.count(node)) return;
        visited.insert(node);

        for (Node* parent : node->parents) {
            topo_sort(parent, visited, topo);
        }

        topo.push_back(node); 
    }
    

private:
    std::vector<Node*> graph_;
    std::vector<Linear> linear_;
    std::vector<Node*> topo_order_;
    std::vector<std::vector<OwnTensor::Tensor>> parameters_;
};