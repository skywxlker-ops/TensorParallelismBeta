#pragma once

#include "../Tensor-Implementations/include/TensorLib.h"
#include "../cgadimpl/cgadimpl/include/ad/ag_all.hpp"
#include <memory>
#include <vector>

// Forward declarations to avoid circular include with dtensor.h
class DTensor;
class ProcessGroup;
class DeviceMesh;
class Layout;

namespace Bridge {

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

// ======================================================
// Autograd Integration
// Provides distributed automatic differentiation for DTensor
// ======================================================

namespace Autograd {

// Wrapper combining DTensor with autograd tracking
struct AutogradDTensor {
    std::shared_ptr<DTensor> dtensor;  // Use pointer to avoid incomplete type
    ag::Value value;
    bool requires_grad;
    
    AutogradDTensor(std::shared_ptr<DTensor> dt, ag::Value v, bool req_grad = false)
        : dtensor(dt), value(v), requires_grad(req_grad) {}
};

// Create autograd-tracked DTensor (parameter)
AutogradDTensor create_parameter(
    const std::vector<float>& data,
    const Layout& layout,
    std::shared_ptr<DeviceMesh> mesh,
    std::shared_ptr<ProcessGroup> pg);

// Forward ops with autograd tracking
AutogradDTensor matmul(const AutogradDTensor& A, const AutogradDTensor& B);
AutogradDTensor add(const AutogradDTensor& A, const AutogradDTensor& B);
AutogradDTensor relu(const AutogradDTensor& A);

// Distributed VJP registration helpers
void register_column_parallel_matmul_vjp(
    ag::Value& result,
    const ag::Value& X,
    const ag::Value& W,
    ProcessGroup* pg);
    
void register_row_parallel_matmul_vjp(
    ag::Value& result,
    const ag::Value& X,
    const ag::Value& W,
    ProcessGroup* pg);

}  // namespace Autograd

}  // namespace Bridge