#include "bridge/tensor_ops_bridge.h"
#include "tensor/dtensor.h" // Include full definitions
#include "tensor/layout.h"
#include "tensor/device_mesh.h"
#include "process_group/process_group.h"

#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>

#pragma GCC visibility push(default)

namespace TensorOpsBridge {

using namespace OwnTensor;

// ------------------------------------------------------------
// Helper: Convert Shape → std::vector<int64_t>
// ------------------------------------------------------------
static std::vector<int64_t> toDims(const Shape& s) {
    return s.dims;
}

// ------------------------------------------------------------
// Elementwise Ops (sanity-checked wrappers)
// (These are fine, no changes needed)
// ------------------------------------------------------------
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

Tensor div(const Tensor& A, const Tensor& B) {
    if (A.shape().dims != B.shape().dims)
        throw std::runtime_error("div: shape mismatch between tensors");
    return A / B;
}

// ------------------------------------------------------------
// Local MatMul (2D/3D) using cuBLAS or fallback
// (This is your local matmul, it is fine, no changes needed)
// ------------------------------------------------------------
Tensor matmul(const Tensor& A, const Tensor& B) {
    const auto& a_dims = A.shape().dims;
    const auto& b_dims = B.shape().dims;

    if (a_dims.size() < 2 || b_dims.size() < 2)
        throw std::runtime_error("matmul: both tensors must be at least 2D");

    // --- Case 1: Simple 2D MatMul ---
    if (a_dims.size() == 2 && b_dims.size() == 2) {
        if (A.device().device == OwnTensor::Device::CUDA &&
            B.device().device == OwnTensor::Device::CUDA) {
            
            // A is [M, K], B is [K, N]
            int M = a_dims[0];
            int K = a_dims[1];
            int N = b_dims[1];
            
            if (a_dims[1] != b_dims[0]) {
                 throw std::runtime_error("matmul: shape mismatch (K dimensions)");
            }

            Tensor C(Shape{{M, N}},
                     TensorOptions()
                         .with_device(A.device())
                         .with_dtype(A.dtype()));

            cublasHandle_t handle;
            cublasCreate(&handle);

            const float alpha = 1.0f;
            const float beta = 0.0f;

            // Note: cuBLAS is column-major.
            // (A[M,K] * B[K,N]) = C[M,N]
            // In Col-major: C_col[N,M] = B_col[N,K] * A_col[K,M]
            // We are passing in row-major pointers, so we reverse the order
            // and swap M/N.
            cublasStatus_t status = cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data<float>(), N,
                A.data<float>(), K,
                &beta,
                C.data<float>(), N);

            cublasDestroy(handle);

            if (status != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("cuBLAS sgemm failed");

            return C;
        }

        // CPU fallback
        return OwnTensor::matmul(A, B);
    }

    // --- Case 2: Batched MatMul (3D) ---
    if (a_dims.size() == 3 && b_dims.size() == 3) {
        return OwnTensor::matmul(A, B);
    }

    // --- Fallback ---
    return OwnTensor::matmul(A, B);
}


// =============================================================
// ✅ NEW UTILITY FUNCTION
// =============================================================
// Creates a new, REPLICATED DTensor from host data.
DTensor from_data(
    const std::vector<float>& host_data,
    const std::vector<int>& shape,
    std::shared_ptr<DeviceMesh> mesh,
    std::shared_ptr<ProcessGroup> pg) 
{
    // 1. Create a new DTensor
    DTensor out(mesh, pg);

    // 2. Define a REPLICATED layout for this new tensor
    Layout replicated_layout(
        mesh,
        shape,                 // The global shape is the shape of the data
        ShardingType::REPLICATED
    );

    // 3. Set the data using the new layout
    out.setData(host_data, replicated_layout);
    
    return out;
}


// =============================================================
// ❌ REMOVED FUNCTION
// =============================================================
//
// DTensor matmul_distributed(...) has been REMOVED.
// All its logic is now inside `DTensor::matmul` in `tensor/dtensor.cpp`.
// This fixes all your compile errors.
//
// =============================================================


}  // namespace TensorOpsBridge

#pragma GCC visibility pop