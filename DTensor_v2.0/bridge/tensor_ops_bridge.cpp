#include "tensor_ops_bridge.h"
#include <iostream>
#include <stdexcept>

namespace TensorOpsBridge {

using namespace OwnTensor;

// --- Helper to safely get shape dimensions ---
static std::vector<int64_t> toDims(const Shape& s) {
    return s.dims;
}

// --- MatMul ---
OwnTensor::Tensor matmul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B) {
    auto a_dims = toDims(A.shape());
    auto b_dims = toDims(B.shape());

    if (a_dims.size() != 2 || b_dims.size() != 2)
        throw std::runtime_error("matmul: only 2D tensors supported.");

    if (a_dims[1] != b_dims[0])
        throw std::runtime_error("matmul: incompatible shapes (" +
                                 std::to_string(a_dims[0]) + "x" + std::to_string(a_dims[1]) +
                                 ") x (" + std::to_string(b_dims[0]) + "x" +
                                 std::to_string(b_dims[1]) + ")");

    return OwnTensor::matmul(A, B);
}

// --- Add ---
OwnTensor::Tensor add(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B) {
    if (A.shape().dims != B.shape().dims)
        throw std::runtime_error("add: shape mismatch between tensors");
    return A + B;
}

// --- Subtract ---
OwnTensor::Tensor sub(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B) {
    if (A.shape().dims != B.shape().dims)
        throw std::runtime_error("sub: shape mismatch between tensors");
    return A - B;
}

// --- Multiply ---
OwnTensor::Tensor mul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B) {
    if (A.shape().dims != B.shape().dims)
        throw std::runtime_error("mul: shape mismatch between tensors");
    return A * B;
}

// --- Divide ---
OwnTensor::Tensor div(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B) {
    if (A.shape().dims != B.shape().dims)
        throw std::runtime_error("div: shape mismatch between tensors");
    return A / B;
}

}  // namespace TensorOpsBridge
