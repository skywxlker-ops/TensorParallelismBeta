#pragma once

#include "../Tensor-Implementations/include/TensorLib.h"
#include <memory>
#include <vector>


class DTensor;
class ProcessGroupNCCL;
class DeviceMesh;

namespace TensorOpsBridge {

using namespace OwnTensor;


OwnTensor::Tensor matmul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor add(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor sub(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor mul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor mul(const OwnTensor::Tensor& A, float scalar);  // Scalar multiplication
OwnTensor::Tensor div(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);

DTensor from_data(
    const std::vector<float>& host_data,
    const std::vector<int64_t>& shape,
    std::shared_ptr<DeviceMesh> device_mesh,
    std::shared_ptr<ProcessGroupNCCL> pg);

} 