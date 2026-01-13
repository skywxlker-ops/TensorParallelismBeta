#include "bridge/bridge.h"
#include "tensor/dtensor.h"
#include "tensor/layout.h"
#include "tensor/device_mesh.h"
// ProcessGroupNCCL is now included via dtensor.h -> ProcessGroupNCCL.h

#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include "../Tensor-Implementations/include/autograd/operations/ActivationOps.h"
#include "../Tensor-Implementations/include/autograd/operations/LossOps.h"

#pragma GCC visibility push(default)

namespace Bridge {

using namespace OwnTensor;

// ------------------------------------------------------------
// Helper: Convert Shape → std::vector<int64_t>
// ------------------------------------------------------------
static std::vector<int64_t> toDims(const Shape& s) {
    return s.dims;
}

// =============================================================================
// BASIC TENSOR OPERATIONS
// =============================================================================

Tensor add(const Tensor& A, const Tensor& B) {
    if (A.shape().dims != B.shape().dims)
        throw std::runtime_error("add: shape mismatch between tensors");
    return A + B;
}

Tensor sub(const Tensor& A, const Tensor& B) {
    if (A.shape().dims != B.shape().dims)
        throw std::runtime_error("sub: shape mismatch between tensors");
    return A - B;
}

Tensor mul(const Tensor& A, const Tensor& B) {
    if (A.shape().dims != B.shape().dims)
        throw std::runtime_error("mul: shape mismatch between tensors");
    return A * B;
}

// Scalar multiplication
Tensor mul(const Tensor& A, float scalar) {
    return A * scalar;
}

Tensor div(const Tensor& A, const Tensor& B) {
    if (A.shape().dims != B.shape().dims)
        throw std::runtime_error("div: shape mismatch between tensors");
    return A / B;
}

// ------------------------------------------------------------
// Local MatMul (2D/3D) using Tensor-Implementations
// ------------------------------------------------------------
Tensor matmul(const Tensor& A, const Tensor& B) {
    const auto& a_dims = A.shape().dims;
    const auto& b_dims = B.shape().dims;

    if (a_dims.size() < 2 || b_dims.size() < 2)
        throw std::runtime_error("matmul: both tensors must be at least 2D");

    // Use Tensor-Implementations for all cases
    return OwnTensor::matmul(A, B);
}


// =============================================================
// UTILITY FUNCTION
// =============================================================
// Creates a new, REPLICATED DTensor from host data.
DTensor from_data(
    const std::vector<float>& host_data,
    const std::vector<int64_t>& shape,
    std::shared_ptr<DeviceMesh> mesh,
    std::shared_ptr<ProcessGroupNCCL> pg) 
{
    // 1. Create a new DTensor
    DTensor out(mesh, pg);

    // 2. Define a REPLICATED layout for this new tensor
    Layout replicated_layout = Layout::replicated(*mesh, shape);

    // 3. Set the data using the new layout
    out.setData(host_data, replicated_layout);
    
    return out;
}


// =============================================================================
// AUTOGRAD OPERATIONS
// =============================================================================

namespace autograd {

Tensor matmul(const Tensor& A, const Tensor& B) {
    // Use autograd-aware matmul if either requires_grad
    if (A.requires_grad() || B.requires_grad()) {
        return OwnTensor::autograd::matmul(A, B);
    }
    // Fall back to regular matmul
    return OwnTensor::matmul(A, B);
}

Tensor add(const Tensor& A, const Tensor& B) {
    if (A.requires_grad() || B.requires_grad()) {
        return OwnTensor::autograd::add(A, B);
    }
    return A + B;
}

Tensor sub(const Tensor& A, const Tensor& B) {
    if (A.requires_grad() || B.requires_grad()) {
        return OwnTensor::autograd::sub(A, B);
    }
    return A - B;
}

Tensor mul(const Tensor& A, const Tensor& B) {
    if (A.requires_grad() || B.requires_grad()) {
        return OwnTensor::autograd::mul(A, B);
    }
    return A * B;
}

Tensor div(const Tensor& A, const Tensor& B) {
    if (A.requires_grad() || B.requires_grad()) {
        return OwnTensor::autograd::div(A, B);
    }
    return A / B;
}

Tensor relu(const Tensor& x) {
    return OwnTensor::autograd::relu(x);
}

Tensor mse_loss(const Tensor& predictions, const Tensor& targets) {
    return OwnTensor::autograd::mse_loss(predictions, targets);
}

void backward(Tensor& output, const Tensor* grad_output) {
    OwnTensor::autograd::backward(output, grad_output);
}

} // namespace autograd

} // namespace Bridge

#pragma GCC visibility pop
