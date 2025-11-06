// #include "tensor_ops_bridge.h"
// #include <iostream>
// #include <stdexcept>

// namespace TensorOpsBridge {

// using namespace OwnTensor;

// // --- Helper to safely get shape dimensions ---
// static std::vector<int64_t> toDims(const Shape& s) {
//     return s.dims;
// }

// // --- MatMul ---
// OwnTensor::Tensor matmul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B) {
//     auto a_dims = toDims(A.shape());
//     auto b_dims = toDims(B.shape());

//     if (a_dims.size() != 2 || b_dims.size() != 2)
//         throw std::runtime_error("matmul: only 2D tensors supported.");

//     if (a_dims[1] != b_dims[0])
//         throw std::runtime_error("matmul: incompatible shapes (" +
//                                  std::to_string(a_dims[0]) + "x" + std::to_string(a_dims[1]) +
//                                  ") x (" + std::to_string(b_dims[0]) + "x" +
//                                  std::to_string(b_dims[1]) + ")");

//     return OwnTensor::matmul(A, B);
// }

// // --- Add ---
// OwnTensor::Tensor add(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B) {
//     if (A.shape().dims != B.shape().dims)
//         throw std::runtime_error("add: shape mismatch between tensors");
//     return A + B;
// }

// // --- Subtract ---
// OwnTensor::Tensor sub(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B) {
//     if (A.shape().dims != B.shape().dims)
//         throw std::runtime_error("sub: shape mismatch between tensors");
//     return A - B;
// }

// // --- Multiply ---
// OwnTensor::Tensor mul(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B) {
//     if (A.shape().dims != B.shape().dims)
//         throw std::runtime_error("mul: shape mismatch between tensors");
//     return A * B;
// }

// // --- Divide ---
// OwnTensor::Tensor div(const OwnTensor::Tensor& A, const OwnTensor::Tensor& B) {
//     if (A.shape().dims != B.shape().dims)
//         throw std::runtime_error("div: shape mismatch between tensors");
//     return A / B;
// }

// }  // namespace TensorOpsBridge


#include "bridge/tensor_ops_bridge.h"
#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace TensorOpsBridge {

using namespace OwnTensor;

// ------------------------------------------------------------
// Helper: Convert Shape → std::vector<int64_t>
// ------------------------------------------------------------
static std::vector<int64_t> toDims(const Shape& s) {
    return s.dims;
}

// ------------------------------------------------------------
// Elementwise Ops (simple sanity checks)
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
// MatMul (supports 2D, 3D, and ND tensors with GPU fallback)
// ------------------------------------------------------------
Tensor matmul(const Tensor& A, const Tensor& B) {
    const auto& a_dims = A.shape().dims;
    const auto& b_dims = B.shape().dims;

    if (a_dims.size() < 2 || b_dims.size() < 2)
        throw std::runtime_error("matmul: both tensors must be at least 2D");

    // --- Case 1: Simple 2D MatMul (prefer cuBLAS on GPU) ---
    if (a_dims.size() == 2 && b_dims.size() == 2) {
        if (A.device().device == Device::CUDA && B.device().device == Device::CUDA) {
            int M = a_dims[0];
            int K = a_dims[1];
            int N = b_dims[1];

            Tensor C(Shape{{M, N}},
                     TensorOptions()
                         .with_device(A.device())
                         .with_dtype(A.dtype()));

            cublasHandle_t handle;
            cublasCreate(&handle);

            const float alpha = 1.0f;
            const float beta = 0.0f;

            cublasStatus_t status = cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data<float>(), N,
                A.data<float>(), K,
                &beta,
                C.data<float>(), N);

            if (status != CUBLAS_STATUS_SUCCESS) {
                cublasDestroy(handle);
                throw std::runtime_error("cuBLAS sgemm failed for 2D matmul");
            }

            cublasDestroy(handle);
            return C;
        }

        // CPU or fallback
        return OwnTensor::matmul(A, B);
    }

    // --- Case 2: Batched (3D) MatMul ---
    if (a_dims.size() == 3 && b_dims.size() == 3 &&
        A.device().device == Device::CUDA && B.device().device == Device::CUDA) {
        int batch = a_dims[0];
        int M = a_dims[1];
        int K = a_dims[2];
        int N = b_dims[2];

        Tensor C(Shape{{batch, M, N}},
                 TensorOptions()
                     .with_device(A.device())
                     .with_dtype(A.dtype()));

        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        int lda = K;
        int ldb = N;
        int ldc = N;

        long long strideA = (long long)M * K;
        long long strideB = (long long)K * N;
        long long strideC = (long long)M * N;

        cublasStatus_t status = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B.data<float>(), ldb, strideB,
            A.data<float>(), lda, strideA,
            &beta,
            C.data<float>(), ldc, strideC,
            batch);

        if (status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            throw std::runtime_error("cuBLAS batched matmul failed");
        }

        cublasDestroy(handle);
        return C;
    }

    // --- Case 3: Generic fallback (TensorLib’s CUDA/CPU path) ---
    return OwnTensor::matmul(A, B);
}

}  // namespace TensorOpsBridge
