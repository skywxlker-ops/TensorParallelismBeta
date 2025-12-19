#include "bridge/tensor_ops_bridge.h"
#include "tensor/dtensor.h" // Include full definitions
#include "tensor/layout.h"
#include "tensor/device_mesh.h"
// ProcessGroupNCCL is now included via dtensor.h -> ProcessGroupNCCL.h

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
// Local MatMul (2D/3D) using cuBLAS or fallback

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
    std::shared_ptr<ProcessGroupNCCL> pg) 
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




// =============================================================
// ✨ AUTOGRAD INTEGRATION (DISABLED - incomplete integration)
// =============================================================
/*
namespace Autograd {

AutogradDTensor create_parameter(
    const std::vector<float>& data,
    const Layout& layout,
    std::shared_ptr<DeviceMesh> mesh,
    std::shared_ptr<ProcessGroup> pg) {
    
    // Create DTensor with the specified layout
    auto dt = std::make_shared<DTensor>(mesh, pg);
    dt->setData(data, layout);
    
    // Wrap local tensor in autograd Value
    ag::Value val = ag::make_tensor(dt->local_tensor(), "param");
    val.node->set_requires_grad(true);
    
    return AutogradDTensor(dt, val, true);
}

AutogradDTensor matmul(const AutogradDTensor& A, const AutogradDTensor& B) {
    // Forward: Use DTensor's sharding-aware matmul
    DTensor result_dt_val = A.dtensor->matmul(*B.dtensor);
    auto result_dt = std::make_shared<DTensor>(std::move(result_dt_val));
    
    // Determine sharding pattern and register appropriate VJP
    const Layout& layout_A = A.dtensor->get_layout();
    const Layout& layout_B = B.dtensor->get_layout();
    
    ag::Value result_val;
    bool needs_grad = A.requires_grad || B.requires_grad;
    
    if (needs_grad) {
        // Use cgadimpl's matmul for autograd tracking
        result_val = ag::matmul(A.value, B.value);
        
        // Register distributed VJP based on sharding pattern
        if (layout_B.sharding_type == ShardingType::SHARDED && layout_B.shard_dim == 1) {
            // Column-parallel: X @ W_col
            register_column_parallel_matmul_vjp(
                result_val, A.value, B.value, A.dtensor->get_pg().get());
        } 
        else if (layout_B.sharding_type == ShardingType::SHARDED && layout_B.shard_dim == 0) {
            // Row-parallel: X @ W_row
            register_row_parallel_matmul_vjp(
                result_val, A.value, B.value, A.dtensor->get_pg().get());
        }
        // else: replicated case, standard VJP is fine
    } else {
        // No gradients needed, just wrap the result
        result_val = ag::make_tensor(result_dt->local_tensor(), "matmul_result");
    }
    
    return AutogradDTensor(result_dt, result_val, needs_grad);
}

AutogradDTensor add(const AutogradDTensor& A, const AutogradDTensor& B) {
    // Forward
    DTensor result_dt_val = A.dtensor->add(*B.dtensor);
    auto result_dt = std::make_shared<DTensor>(std::move(result_dt_val));
    
    // Autograd (standard add VJP works fine for distributed)
    ag::Value result_val;
    bool needs_grad = A.requires_grad || B.requires_grad;
    
    if (needs_grad) {
        result_val = ag::add(A.value, B.value);
    } else {
        result_val = ag::make_tensor(result_dt->local_tensor(), "add_result");
    }
    
    return AutogradDTensor(result_dt, result_val, needs_grad);
}

AutogradDTensor relu(const AutogradDTensor& A) {
    // Forward: element-wise, so just use local tensor
    OwnTensor::Tensor result_local = OwnTensor::relu(A.dtensor->local_tensor(), nullptr);
    
    // Create DTensor with same layout as input
    auto result_dt = std::make_shared<DTensor>(
        A.dtensor->get_mesh(), A.dtensor->get_pg(), 
        result_local, A.dtensor->get_layout());
    
    // Autograd
    ag::Value result_val;
    if (A.requires_grad) {
        result_val = ag::relu(A.value);
    } else {
        result_val = ag::make_tensor(result_local, "relu_result");
    }
    
    return AutogradDTensor(result_dt, result_val, A.requires_grad);
}

void register_column_parallel_matmul_vjp(
    ag::Value& result,
    const ag::Value& X,
    const ag::Value& W,
    ProcessGroup* pg) {
    
    // Custom VJP: Y = X @ W_col_shard
    // grad_X = grad_Y @ W^T (needs AllReduce across column shards)
    // grad_W = X^T @ grad_Y (local computation, result is column-sharded)
    
    auto vjp_fn = [pg](ag::Node* node, const OwnTensor::Tensor& grad_Y) {
        ag::Node* X_node = node->inputs[0].get();
        ag::Node* W_node = node->inputs[1].get();
        
        if (W_node->requires_grad()) {
            // grad_W = X^T @ grad_Y (local)
            OwnTensor::Tensor grad_W = Bridge::matmul(
                OwnTensor::transpose(X_node->value), grad_Y);
            W_node->grad += grad_W;
        }
        
        if (X_node->requires_grad()) {
            // grad_X_local = grad_Y @ W^T
            OwnTensor::Tensor grad_X_local = Bridge::matmul(
                grad_Y, OwnTensor::transpose(W_node->value));
            
            // AllReduce to combine across column shards
            float* grad_data = grad_X_local.data<float>();
            int64_t count = grad_X_local.numel();
            cudaStream_t stream = nullptr; // TODO: use proper stream
            
            pg->allReduce(grad_data, count, ncclFloat32, ncclSum, stream);
            
            X_node->grad += grad_X_local;
        }
    };
    
    ag::register_custom_vjp(result.node, vjp_fn);
}

void register_row_parallel_matmul_vjp(
    ag::Value& result,
    const ag::Value& X,
    const ag::Value& W,
    ProcessGroup* pg) {
    
    // Custom VJP: Y = X @ W_row_shard (result needs AllReduce in forward)
    // grad_X = grad_Y @ W^T (local, stays column-sharded)
    // grad_W = X^T @ grad_Y (local, stays row-sharded)
    
    auto vjp_fn = [pg](ag::Node* node, const OwnTensor::Tensor& grad_Y) {
        ag::Node* X_node = node->inputs[0].get();
        ag::Node* W_node = node->inputs[1].get();
        
        if (W_node->requires_grad()) {
            // grad_W = X^T @ grad_Y (local, row-sharded)
            OwnTensor::Tensor grad_W = Bridge::matmul(
                OwnTensor::transpose(X_node->value), grad_Y);
            W_node->grad += grad_W;
        }
        
        if (X_node->requires_grad()) {
            // grad_X = grad_Y @ W^T (local, column-sharded)
            OwnTensor::Tensor grad_X = Bridge::matmul(
                grad_Y, OwnTensor::transpose(W_node->value));
            // No AllReduce needed, gradient stays column-sharded
            X_node->grad += grad_X;
        }
    };
    
    ag::register_custom_vjp(result.node, vjp_fn);
}

}  // namespace Autograd
*/

}  // namespace TensorOpsBridge

#pragma GCC visibility pop


