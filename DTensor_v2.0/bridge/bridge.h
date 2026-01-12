#pragma once

#include "../Tensor-Implementations/include/TensorLib.h"
#include <memory>
#include <vector>

// Autograd includes
#include "../Tensor-Implementations/include/autograd/Node.h"
#include "../Tensor-Implementations/include/autograd/Variable.h"
#include "../Tensor-Implementations/include/autograd/Engine.h"
#include "../Tensor-Implementations/include/autograd/operations/BinaryOps.h"
#include "../Tensor-Implementations/include/autograd/operations/MatrixOps.h"

class DTensor;
class ProcessGroupNCCL;
class DeviceMesh;

namespace Bridge {

using namespace OwnTensor;

// =============================================================================
// BASIC TENSOR OPERATIONS
// =============================================================================

OwnTensor::Tensor matmul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor add(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor sub(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor mul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor mul(const OwnTensor::Tensor& A, float scalar);
OwnTensor::Tensor div(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);

DTensor from_data(
    const std::vector<float>& host_data,
    const std::vector<int64_t>& shape,
    std::shared_ptr<DeviceMesh> device_mesh,
    std::shared_ptr<ProcessGroupNCCL> pg);

// =============================================================================
// AUTOGRAD OPERATIONS
// =============================================================================

namespace autograd {

/**
 * Autograd-aware matmul. Builds gradient graph if either input requires_grad.
 */
OwnTensor::Tensor matmul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);

/**
 * Autograd-aware elementwise operations.
 */
OwnTensor::Tensor add(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor sub(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor mul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor div(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);

/**
 * Execute backward pass on a tensor.
 */
void backward(OwnTensor::Tensor& output, const OwnTensor::Tensor* grad_output = nullptr);

} // namespace autograd

} // namespace Bridge
