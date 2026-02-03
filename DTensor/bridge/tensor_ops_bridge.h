#pragma once

#include <memory>
#include <vector>
#include "TensorLib.h"


class DTensor;
class ProcessGroup;
class DeviceMesh;

namespace TensorOpsBridge {


OwnTensor::Tensor matmul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor add(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor sub(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor mul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor mul(const OwnTensor::Tensor& A, float scalar);  // Scalar multiplication
OwnTensor::Tensor div(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);

DTensor from_data(
    const std::vector<float>& host_data,
    const std::vector<int>& shape,
    std::shared_ptr<DeviceMesh> device_mesh,
    std::shared_ptr<ProcessGroup> pg);

} 