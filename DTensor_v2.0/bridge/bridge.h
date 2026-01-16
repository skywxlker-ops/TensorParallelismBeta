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
 * Autograd-aware activations.
 */
OwnTensor::Tensor relu(const OwnTensor::Tensor& x);

/**
 * Autograd-aware loss functions.
 */
OwnTensor::Tensor mse_loss(const OwnTensor::Tensor& predictions, const OwnTensor::Tensor& targets);

/**
 * Autograd-aware GeLU activation.
 */
OwnTensor::Tensor gelu(const OwnTensor::Tensor& x);

/**
 * Autograd-aware softmax.
 */
OwnTensor::Tensor softmax(const OwnTensor::Tensor& x, int64_t dim = -1);

/**
 * Autograd-aware categorical cross entropy loss.
 */
OwnTensor::Tensor categorical_cross_entropy(const OwnTensor::Tensor& predictions, const OwnTensor::Tensor& targets);

/**
 * Autograd-aware embedding lookup.
 * @param indices Tensor of token IDs (uint16)
 * @param weight Embedding weight matrix [vocab_size, embedding_dim]
 * @param padding_idx Index to treat as padding (-1 for none)
 */
OwnTensor::Tensor embedding(const OwnTensor::Tensor& indices, const OwnTensor::Tensor& weight, int padding_idx = -1);

/**
 * Execute backward pass on a tensor.
 */
void backward(OwnTensor::Tensor& output, const OwnTensor::Tensor* grad_output = nullptr);

} // namespace autograd

} // namespace Bridge
