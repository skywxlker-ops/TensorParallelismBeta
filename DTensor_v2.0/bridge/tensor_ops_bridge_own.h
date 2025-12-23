#pragma once

#include "..//home/blu-bridge25/Study/Code/Tensor_Parallelism_impl/tenosr_parallelism/TensorParallelismBeta/DTensor_v2.0/Tensor-Implementations/include/TensorLib.h"
#include <memory>
#include <vector>


class DTensor;
class ProcessGroup;
class DeviceMesh;

namespace TensorOpsBridgeCustom {

using namespace OwnTensor;



OwnTensor::Tensor matmul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor add(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor sub(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor mul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor div(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);


DTensor from_data(
    const std::vector<float>& host_data,
    const std::vector<int>& shape,
    std::shared_ptr<DeviceMesh> device_mesh,
    std::shared_ptr<ProcessGroup> pg);

}  
