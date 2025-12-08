#include "bridge/tensor_ops_bridge.h"
#include "tensor/dtensor.h" 
#include "tensor/layout.h"
#include "tensor/device_mesh.h"
#include "tensor/placement.h"
#include "process_group/process_group.h"

#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>

#pragma GCC visibility push(default)

namespace TensorOpsBridge {

using namespace OwnTensor;

static std::vector<int64_t> toDims(const Shape& s) {
    return s.dims;
}


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

Tensor mul(const Tensor& A, float scalar) {
    // Scalar multiplication - multiply all elements by scalar
    return A * scalar;
}

Tensor div(const Tensor& A, const Tensor& B) {
    if (A.shape().dims != B.shape().dims)
        throw std::runtime_error("div: shape mismatch between tensors");
    return A / B;
}



Tensor matmul(const Tensor& A, const Tensor& B) {
    const auto& a_dims = A.shape().dims;
    const auto& b_dims = B.shape().dims;

    if (a_dims.size() < 2 || b_dims.size() < 2)
        throw std::runtime_error("matmul: both tensors must be at least 2D");

 
    if (a_dims.size() == 2 && b_dims.size() == 2) {
        if (A.device().device == OwnTensor::Device::CUDA &&
            B.device().device == OwnTensor::Device::CUDA) {
            
            // A : [M, K], B : [K, N]
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

        
        return OwnTensor::matmul(A, B);
    }

    //  3D 
    if (a_dims.size() == 3 && b_dims.size() == 3) {
        return OwnTensor::matmul(A, B);
    }

    
    return OwnTensor::matmul(A, B);
}


DTensor from_data(
    const std::vector<float>& host_data,
    const std::vector<int>& shape,
    std::shared_ptr<DeviceMesh> device_mesh,
    std::shared_ptr<ProcessGroup> pg) 
{

    DTensor out(device_mesh, pg);

   
    Layout replicated_layout = Layout::replicated(device_mesh, shape);

  
    out.setData(host_data, replicated_layout);
    
    return out;
}



} 

#pragma GCC visibility pop


