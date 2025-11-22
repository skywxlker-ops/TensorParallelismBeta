#pragma once

#include "../Tensor-Implementations/include/TensorLib.h"
#include <memory>
#include <vector>

// Forward declarations to avoid circular include with dtensor.h
class DTensor;
class ProcessGroup;
class DeviceMesh;

namespace TensorOpsBridge {

using namespace OwnTensor;

// ======================================================
// Tensor Operations Bridge
// Provides DTensor with unified LOCAL TensorLib interfaces
// ======================================================

// ---------------- Local Ops ----------------
// These all operate on the underlying OwnTensor::Tensor

OwnTensor::Tensor matmul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor add(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor sub(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor mul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor div(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);

// ---------------- Utility ----------------

// Creates a new, REPLICATED DTensor from host data.
DTensor from_data(
    const std::vector<float>& host_data,
    const std::vector<int>& shape,
    std::shared_ptr<DeviceMesh> mesh,
    std::shared_ptr<ProcessGroup> pg);

}  // namespace TensorOpsBridge