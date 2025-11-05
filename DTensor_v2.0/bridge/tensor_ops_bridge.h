#pragma once

#include "TensorLib.h"
#include "ops/TensorOps.h"

namespace TensorOpsBridge {

using namespace OwnTensor;

// ======================================================
// Tensor Operations Bridge
// Provides DTensor with unified TensorLib interfaces
// ======================================================

// Matrix Multiplication
OwnTensor::Tensor matmul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);

// Element-wise Ops
OwnTensor::Tensor add(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor sub(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor mul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);
OwnTensor::Tensor div(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B);

}  // namespace TensorOpsBridge
