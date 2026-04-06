#pragma once
// ---------------------------------------------------------------------------
// CublasLtMatmul.h
//
// Lightweight cuBLASLt wrapper with heuristic algorithm selection.
// Provides a single function for FP32 row-major matmul:
//
//   C = A @ B   where A: [M, K], B: [K, N], C: [M, N]
//
// On the FIRST call for a given (M, K, N) shape, cuBLASLt runs heuristic
// search to find the fastest algorithm. The result is cached so subsequent
// calls with the same shape skip the search entirely.
//
// Usage (e.g. for LM head in gpt2_cp_test.cpp):
//
//   #include "dnn/CublasLtMatmul.h"
//   ...
//   // Replaces:  Tensor logits = lm_head->forward(x);
//   // With:
//   Tensor logits = cublaslt_matmul(x_2d, lm_head->weight);
//
// ---------------------------------------------------------------------------

#include "core/Tensor.h"

namespace OwnTensor {
namespace dnn {

/// Row-major FP32 matmul using cuBLASLt with heuristic algorithm selection.
/// A: [M, K]  B: [K, N]  => C: [M, N]
///
/// Both A and B must be contiguous, FP32, on the same CUDA device.
/// Returns a new Tensor C with requires_grad=false.
Tensor cublaslt_matmul(const Tensor& A, const Tensor& B);

} // namespace dnn
} // namespace OwnTensor
