// Minimal Bridge stubs for compilation without autograd
#include "bridge/tensor_ops_bridge.h"

namespace Bridge {

OwnTensor::Tensor add(const OwnTensor::Tensor& a, const OwnTensor::Tensor& b) {
    return a + b;  // Uses operator+ overload
}

OwnTensor::Tensor sub(const OwnTensor::Tensor& a, const OwnTensor::Tensor& b) {
    return a - b;  // Uses operator- overload
}

OwnTensor::Tensor mul(const OwnTensor::Tensor& a, const OwnTensor::Tensor& b) {
    return a * b;  // Uses operator* overload
}

OwnTensor::Tensor div(const OwnTensor::Tensor& a, const OwnTensor::Tensor& b) {
    return a / b;  // Uses operator/ overload
}

OwnTensor::Tensor matmul(const OwnTensor::Tensor& a, const OwnTensor::Tensor& b) {
    return OwnTensor::matmul(a, b);  // Uses free function
}

} // namespace Bridge
